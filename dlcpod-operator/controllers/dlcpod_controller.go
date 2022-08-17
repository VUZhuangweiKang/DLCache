/*
Copyright 2022.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controllers

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"sort"
	"sync"
	"time"

	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/bigkevmcd/go-configparser"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"

	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/VUZhuangweiKang/DLCache/tree/main/dlcpod-operator/api/v1alpha1"
	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	APIVersion    = "core/v1"
	PodKind       = "Pod"
	SecretKind    = "Secret"
	ConfigMapKind = "ConfigMap"
	ClientImage   = "zhuangeweikang/dlcpod:client"
)

// DLCPodReconciler reconciles a DLCPod object
type DLCPodReconciler struct {
	client.Client
	Log    logr.Logger
	Scheme *runtime.Scheme
}

func checkErr(err error) {
	if err != nil {
		panic(err.Error())
	}
}

//+kubebuilder:rbac:groups=docgroup.com,resources=dlcpods,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=docgroup.com,resources=dlcpods/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=docgroup.com,resources=dlcpods/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
func (r *DLCPodReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := r.Log.WithValues("dlcpod", req.NamespacedName)
	log.Info("start reconciling")
	var dlcpod v1alpha1.DLCPod
	err := r.Get(ctx, req.NamespacedName, &dlcpod)
	if err != nil {
		if k8serrors.IsNotFound(err) {
			// Request object not found, could have been deleted after reconcile request.
			// Owned objects are automatically garbage collected. For additional cleanup logic use finalizers.
			// Return and don't requeue
			return ctrl.Result{}, nil
		}
		// Error reading the object - requeue the request.
		return ctrl.Result{}, err
	}

	if dlcpod.DeletionTimestamp != nil {
		return ctrl.Result{}, err
	}

	pod := &corev1.Pod{}
	if err := r.Get(ctx, req.NamespacedName, pod); err != nil && k8serrors.IsNotFound(err) {
		schedule := r.scheduler(ctx, dlcpod)
		if err != nil {
			return ctrl.Result{}, err
		}
		dlcpod.Spec.NodeSequence = schedule

		// create pod
		pod, err := r.createPod(ctx, dlcpod)
		var selectNode string
		if len(dlcpod.Spec.NodeSelector) == 0 {
			selectNode = schedule[0]
		} else {
			selectNode = pod.Spec.NodeName
		}
		pod.Spec.NodeName = selectNode
		if err != nil {
			log.Error(err, fmt.Sprintf("error in creating pod %s: %s.", pod.Name, err.Error()))
			return ctrl.Result{}, err
		} else {
			if err := r.Create(ctx, &pod, &client.CreateOptions{}); err != nil {
				return ctrl.Result{}, err
			}
			// associate Annotations
			data, _ := json.Marshal(dlcpod.Spec)
			if dlcpod.Annotations != nil {
				dlcpod.Annotations["spec"] = string(data)
			} else {
				dlcpod.Annotations = map[string]string{"spec": string(data)}
			}
			if err := r.Update(ctx, &dlcpod, &client.UpdateOptions{}); err != nil {
				return ctrl.Result{}, nil
			}

			// create configmap
			configmap := &corev1.ConfigMap{}
			if err := r.Get(ctx, req.NamespacedName, configmap); err != nil && k8serrors.IsNotFound(err) {
				configmap, err := r.createConfigMap(ctx, dlcpod)
				if err != nil {
					log.Error(err, fmt.Sprintf("error in creating configmap %s: %s.", configmap.Name, err.Error()))
					return ctrl.Result{}, err
				} else {
					if err := r.Create(ctx, &configmap, &client.CreateOptions{}); err != nil {
						return ctrl.Result{}, err
					}
				}
			}
			return ctrl.Result{}, nil
		}
	}

	// update associated resources
	oldspec := v1alpha1.DLCPodSpec{}
	if err := json.Unmarshal([]byte(dlcpod.Annotations["spec"]), &oldspec); err != nil {
		return ctrl.Result{}, nil
	}
	if !reflect.DeepEqual(dlcpod.Spec, oldspec) {
		// update configmap
		schedule := r.scheduler(ctx, dlcpod)
		dlcpod.Spec.NodeSequence = schedule
		oldpod := corev1.Pod{}
		if err := r.Get(ctx, req.NamespacedName, &oldpod); err != nil {
			log.Error(err, fmt.Sprintf("error in getting pod %s", req.Name))
			return ctrl.Result{}, err
		}

		// delete the old pod
		if err := r.Delete(ctx, &oldpod, &client.DeleteOptions{}); err != nil {
			log.Error(err, fmt.Sprintf("failed to delete old pod %s", dlcpod.Name))
			return ctrl.Result{}, err
		}

		// create a new pod
		newpod, _ := r.createPod(ctx, dlcpod)
		var selectNode string
		if len(dlcpod.Spec.NodeSelector) == 0 {
			selectNode = schedule[0]
		} else {
			selectNode = newpod.Spec.NodeName
		}
		newpod.Spec.NodeName = selectNode
		if err := r.Create(ctx, &newpod, &client.CreateOptions{}); err != nil {
			log.Error(err, fmt.Sprintf("error in creating pod %s: %s.", pod.Name, err.Error()))
			return ctrl.Result{}, err
		}

		// create configmap
		newconfigmap, _ := r.createConfigMap(ctx, dlcpod)
		if err := r.Update(ctx, &newconfigmap, &client.UpdateOptions{}); err != nil {
			log.Error(err, fmt.Sprintf("error in updating configmap %s", req.Name))
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, nil
	}
	return ctrl.Result{}, nil
}

func (r *DLCPodReconciler) scheduler(ctx context.Context, dlcpod v1alpha1.DLCPod) []string {
	spec := dlcpod.Spec
	nodes := &corev1.NodeList{}
	err := r.List(ctx, nodes, &client.ListOptions{})
	checkErr(err)
	nodeAddresses := []string{}
	for _, node := range nodes.Items {
		nodeAddresses = append(nodeAddresses, node.Status.Addresses[0].Address)
	}

	// get dataset Etags (MD5 Hash) from S3
	secret := corev1.Secret{}
	err = r.Get(ctx, types.NamespacedName{Namespace: dlcpod.Namespace, Name: spec.Secret.Name}, &secret)
	checkErr(err)
	awsctx, cancel := context.WithTimeout(context.TODO(), 20*time.Second)
	defer cancel()
	err = os.WriteFile(fmt.Sprintf("/tmp/%s", spec.Secret.Name), secret.Data["client.conf"], 0644)
	checkErr(err)
	parser, _ := configparser.NewConfigParserFromFile(fmt.Sprintf("/tmp/%s", spec.Secret.Name))
	key, err := parser.Get("AWS", "AWS_ACCESS_KEY_ID")
	checkErr(err)
	secretId, err := parser.Get("AWS", "AWS_SECRET_ACCESS_KEY")
	checkErr(err)

	sessionToken, err := parser.Get("AWS", "AWS_SESSION_TOKEN")
	if err != nil {
		sessionToken = "" // session token is set to None, may cause problem
	}
	cfg, err := awsconfig.LoadDefaultConfig(awsctx, awsconfig.WithCredentialsProvider(
		credentials.NewStaticCredentialsProvider(key, secretId, sessionToken),
	))
	checkErr(err)
	s3client := s3.NewFromConfig(cfg)

	etags := []string{}
	for _, job := range spec.Jobs {
		for _, prefix := range job.DataSource.Keys {
			var wg sync.WaitGroup
			worker := func(wg *sync.WaitGroup, page *s3.ListObjectsV2Output) {
				defer wg.Done()
				for _, obj := range page.Contents {
					etags = append(etags, *obj.ETag)
				}
			}
			paginator := s3.NewListObjectsV2Paginator(s3client, &s3.ListObjectsV2Input{
				Bucket: &job.DataSource.Bucket,
				Prefix: &prefix,
			})
			for paginator.HasMorePages() {
				page, err := paginator.NextPage(context.TODO())
				checkErr(err)
				wg.Add(1)
				go worker(&wg, page)
			}
			wg.Wait()
		}
	}

	// find existing keys on NFS
	mongoctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	// TODO: update the mongodb URI
	mongoclient, err := mongo.Connect(mongoctx, options.Client().ApplyURI("mongodb://docgroup:docgroup@mongo:27017"))
	checkErr(err)
	collection := mongoclient.Database("DLCache").Collection("Datasets")
	filterCursor, err := collection.Find(mongoctx, bson.M{"ETag": bson.M{"$in": etags}})
	checkErr(err)
	var existingETags []bson.M
	err = filterCursor.All(mongoctx, &existingETags)
	checkErr(err)

	// generate scheduling policy
	weights := map[string]float64{}
	allocatableResource := map[string]map[string]int64{}
	for _, node := range nodes.Items {
		allocatable := node.Status.Allocatable
		gpus := allocatable["nvidia.com/gpu"]
		allocatableGPU, ok := gpus.AsInt64()
		if !ok {
			allocatableGPU = 0
		}
		allocatableResource[node.Status.Addresses[0].Address] = map[string]int64{
			"storage": allocatable.Storage().Value(),
			"memory":  allocatable.Memory().Value(),
			"cpu":     allocatable.Cpu().Value(),
			"gpu":     allocatableGPU,
		}
	}
	var totalCapacity float64
	for node := range allocatableResource {
		totalCapacity += float64(allocatableResource[node]["storage"])
	}

	/*
		weights[node] = (#existingETagsOnNode/#totalETags) + (1-#totalExistingETags/#totalETags)*(allocataleNodeSpace/totalCapacity)
	*/
	if len(existingETags) > 0 {
		for _, item := range existingETags {
			if val, ok := weights[item["Location"].(string)]; ok {
				weights[item["Location"].(string)] = val + 1
			} else {
				weights[item["Location"].(string)] = 1
			}
		}
		for node := range weights {
			weights[node] = float64(weights[node] / float64(len(etags)))
		}
		r := 1 - (float64(len(existingETags)) / float64(len(etags)))
		for _, node := range nodes.Items {
			node := node.Status.Addresses[0].Address
			freeDisk := allocatableResource[node]["disk"]
			if _, ok := weights[node]; ok {
				weights[node] += r * float64(freeDisk) / totalCapacity
			} else {
				weights[node] = float64(freeDisk) / totalCapacity
			}
		}
	} else {
		for _, node := range nodes.Items {
			weights[node.Status.Addresses[0].Address] = float64(allocatableResource[node.Name]["disk"]) / totalCapacity
		}
	}

	/*
		We assign the job to the node that has maximum free space and sufficien GPU.
		Place dataset on other nodes sorted by free space.
		TODO: how to rebalance data
	*/
	var requestGPU int64 = 0
	for _, job := range spec.Jobs {
		gpu := job.Resources.Requests["nvidia.com/gpu"]
		requestGPU += gpu.Value()
	}
	sort.SliceStable(nodeAddresses, func(i, j int) bool {
		return weights[nodeAddresses[i]] > weights[nodeAddresses[j]]
	})
	schedule := []string{}
	i := 0
	for ; i < len(nodeAddresses); i++ {
		node := nodeAddresses[i]
		if allocatableResource[node]["gpu"] >= requestGPU {
			schedule = append(schedule, node)
			break
		}
	}
	nodeAddresses = append(nodeAddresses[:i], nodeAddresses[i+1:]...)
	return append(schedule, nodeAddresses...)
}

func (r *DLCPodReconciler) createPod(ctx context.Context, dlcpod v1alpha1.DLCPod) (corev1.Pod, error) {
	pod := corev1.Pod{}
	spec := dlcpod.Spec
	volumes := []corev1.Volume{
		{
			Name:         "secret",
			VolumeSource: corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{SecretName: spec.Secret.Name}},
		},
		{
			Name:         "jobsmeta",
			VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{LocalObjectReference: corev1.LocalObjectReference{Name: dlcpod.Name}}},
		},
		{
			Name:         "share",
			VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
		},
		{
			Name:         "shmem",
			VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{Path: "/dev/shm"}},
		},
		{
			Name:         "runtime",
			VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{Medium: corev1.StorageMediumMemory}},
		},
	}

	volumes = append(volumes, spec.Volumes...)
	vol_mounts := []corev1.VolumeMount{
		{Name: "secret", MountPath: "/secret"},
		{Name: "jobsmeta", MountPath: "/jobsmeta"},
		{Name: "share", MountPath: "/share"},
		{Name: "runtime", MountPath: "/runtime"},
		{Name: "shmem", MountPath: "/dev/shm"},
	}
	nodes := &corev1.NodeList{}
	err := r.List(ctx, nodes, &client.ListOptions{})
	checkErr(err)
	var jobNode corev1.Node
	for _, node := range nodes.Items {
		nodeip := node.Status.Addresses[0].Address
		if nodeip == dlcpod.Spec.NodeSequence[0] {
			jobNode = node
		}
		name := fmt.Sprintf("/%s", nodeip)
		volumes = append(volumes, corev1.Volume{
			Name: name,
			VolumeSource: corev1.VolumeSource{NFS: &corev1.NFSVolumeSource{
				Server:   nodeip,
				Path:     "/nfs_storage",
				ReadOnly: false,
			}},
		})
		vol_mounts = append(vol_mounts, corev1.VolumeMount{
			Name:      name,
			MountPath: name,
		})
	}

	capacity := jobNode.Status.Capacity
	gpus := capacity["nvidia.com/gpu"]
	capacityGPU, ok := gpus.AsInt64()
	if !ok {
		capacityGPU = 0
	}
	var containers []corev1.Container
	for _, job := range spec.Jobs {
		env := job.Env
		env = append(env, corev1.EnvVar{Name: "JOBNAME", Value: job.Name})
		cpuLimit := capacity.Cpu().MilliValue() / capacityGPU / int64(len(spec.Jobs))
		memoryLimit := capacity.Memory().MilliValue() / capacityGPU / int64(len(spec.Jobs))
		storageLimit := capacity.Storage().MilliValue() / capacityGPU / int64(len(spec.Jobs))
		gpuLimit := job.Resources.Limits["nvidia.com/gpu"]
		container := corev1.Container{
			Name:            job.Name,
			Image:           job.Image,
			ImagePullPolicy: job.ImagePullPolicy,
			WorkingDir:      job.WorkingDir,
			Env:             env,
			EnvFrom:         job.EnvFrom,
			Command:         job.Command,
			VolumeMounts:    vol_mounts,
			Ports:           job.Ports,
			Lifecycle:       job.Lifecycle,
			Resources: corev1.ResourceRequirements{
				Limits: corev1.ResourceList{
					"cpu":            *resource.NewMilliQuantity(cpuLimit, resource.DecimalSI),
					"memory":         *resource.NewMilliQuantity(memoryLimit, resource.DecimalSI),
					"storage":        *resource.NewMilliQuantity(storageLimit, resource.DecimalSI),
					"nvidia.com/gpu": *resource.NewQuantity(gpuLimit.Value(), resource.DecimalSI),
				},
				Requests: job.Resources.Requests,
			},
			TTY:   job.TTY,
			Stdin: job.Stdin,
		}
		containers = append(containers, container)
	}

	container := corev1.Container{
		Name:            "client",
		Image:           ClientImage,
		ImagePullPolicy: corev1.PullAlways,
		WorkingDir:      "/app",
		// Command:         []string{"python3", "client.py"},
		Env: []corev1.EnvVar{{
			Name:      "NODE_IP",
			ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "status.hostIP"}}}},
		VolumeMounts: vol_mounts,
		TTY:          true,
		Stdin:        true,
	}
	containers = append(containers, container)

	pod = corev1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       PodKind,
			APIVersion: APIVersion,
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      dlcpod.Name,
			Namespace: dlcpod.Namespace,
			// Annotations: map[string]string{"k8s.v1.cni.cncf.io/networks": "macvlan-conf"},
		},
		Spec: corev1.PodSpec{
			Volumes:       volumes,
			Containers:    containers,
			RestartPolicy: corev1.RestartPolicyNever,
			NodeSelector:  spec.NodeSelector,
			NodeName:      spec.NodeName,
			HostNetwork:   spec.HostNetwork,
		},
	}

	err = ctrl.SetControllerReference(&dlcpod, &pod, r.Scheme)
	checkErr(err)
	return pod, nil
}

func (r *DLCPodReconciler) createConfigMap(ctx context.Context, dlcpod v1alpha1.DLCPod) (corev1.ConfigMap, error) {
	configmap := corev1.ConfigMap{
		TypeMeta:   metav1.TypeMeta{Kind: ConfigMapKind, APIVersion: APIVersion},
		ObjectMeta: metav1.ObjectMeta{Name: dlcpod.Name, Namespace: dlcpod.Namespace},
		Data:       map[string]string{},
	}
	spec := dlcpod.Spec
	for _, job := range spec.Jobs {
		jobinfo := map[string]interface{}{
			"name":       job.Name,
			"dataSource": job.DataSource,
			"weights":    dlcpod.Spec.NodeSequence,
		}
		if job.ConfigurationsFromConfigMap.Name != "" {
			var qos_config corev1.ConfigMap
			err := r.Get(ctx, types.NamespacedName{Name: job.ConfigurationsFromConfigMap.Name, Namespace: dlcpod.Namespace}, &qos_config)
			checkErr(err)
			jobinfo["qos"] = qos_config.Data
		} else {
			qos_data := make(map[string]interface{})
			v := reflect.ValueOf(job.QoS)
			t := v.Type()
			for i := 0; i < v.NumField(); i++ {
				qos_data[t.Field(i).Name] = v.Field(i).Interface()
			}
			jobinfo["qos"] = qos_data
		}
		byte_arr, _ := json.Marshal(jobinfo)
		configmap.Data[fmt.Sprintf("%s.json", job.Name)] = string(byte_arr)
	}

	err := ctrl.SetControllerReference(&dlcpod, &configmap, r.Scheme)
	checkErr(err)
	return configmap, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *DLCPodReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha1.DLCPod{}).
		Owns(&corev1.ConfigMap{}).
		Owns(&corev1.Pod{}).
		Complete(r)
}
