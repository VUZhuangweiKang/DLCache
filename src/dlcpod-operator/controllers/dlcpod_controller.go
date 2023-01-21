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
	"io/ioutil"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"

	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/VUZhuangweiKang/DLCache/tree/main/src/dlcpod-operator/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	APIVersion    = "core/v1"
	PodKind       = "Pod"
	SecretKind    = "Secret"
	ConfigMapKind = "ConfigMap"
	ClientImage   = "zhuangweikang/dlcache-dev:client"
)

// DLCPodReconciler reconciles a DLCPod object
type DLCPodReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

func checkErr(err error) {
	if err != nil {
		panic(err.Error())
	}
}

func contains(s []string, str string) bool {
	for _, v := range s {
		if v == str {
			return true
		}
	}
	return false
}

//+kubebuilder:rbac:groups=docgroup.com,resources=dlcpods,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=docgroup.com,resources=dlcpods/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=docgroup.com,resources=dlcpods/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
func (r *DLCPodReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)
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
		schedule := r.scheduler(ctx, &dlcpod)
		addresses := []string{}
		hostnames := []string{}
		for i := 0; i < len(schedule); i++ {
			addresses = append(addresses, schedule[i][0])
			hostnames = append(hostnames, schedule[i][1])
		}
		dlcpod.Spec.NodeSequence = addresses

		// create pod
		pod, err := r.createPod(ctx, &dlcpod)
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
				return ctrl.Result{}, err
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
						log.Error(err, fmt.Sprintf("error in creating configmap %s: %s.", configmap.Name, err.Error()))
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
		schedule := r.scheduler(ctx, &dlcpod)
		addresses := []string{}
		hostnames := []string{}
		for i := 0; i < len(schedule); i++ {
			addresses = append(addresses, schedule[i][0])
			hostnames = append(hostnames, schedule[i][1])
		}
		dlcpod.Spec.NodeSequence = addresses
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
		newpod, _ := r.createPod(ctx, &dlcpod)
		var selectNode string
		if len(dlcpod.Spec.NodeSelector) == 0 {
			selectNode = hostnames[0]
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
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, nil
	}
	return ctrl.Result{}, nil
}

func (r *DLCPodReconciler) filterNodes(ctx context.Context) []corev1.Node {
	allnodes := &corev1.NodeList{}
	err := r.List(ctx, allnodes, &client.ListOptions{})
	checkErr(err)
	nodes := []corev1.Node{}
	activeNodeStatus := map[corev1.NodeConditionType]corev1.ConditionStatus{
		"NetworkUnavailable": "False",
		"PIDPressure":        "False",
		"DiskPressure":       "False",
		"MemoryPressure":     "False",
		"Ready":              "True",
	}
	for _, node := range allnodes.Items {
		pass_status_check := true
		for _, condition := range node.Status.Conditions {
			if condition.Status != activeNodeStatus[condition.Type] {
				pass_status_check = false
			}
		}
		if pass_status_check {
			nodes = append(nodes, node)
		}
	}
	return nodes
}

func (r *DLCPodReconciler) scheduler(ctx context.Context, dlcpod *v1alpha1.DLCPod) [][]string {
	spec := dlcpod.Spec
	nodes := r.filterNodes(ctx)
	nodeAddresses := []string{}
	addressHostMap := map[string]string{}
	for _, node := range nodes {
		addr := node.Status.Addresses[0].Address
		hostname := node.Status.Addresses[1].Address
		nodeAddresses = append(nodeAddresses, addr)
		addressHostMap[addr] = hostname
	}

	// get dataset Etags (MD5 Hash) from S3
	var secret corev1.Secret
	err := r.Get(ctx, types.NamespacedName{Namespace: dlcpod.Namespace, Name: spec.Secret.Name}, &secret)
	checkErr(err)

	// connect to S3
	awsctx, cancel := context.WithTimeout(context.TODO(), 20*time.Second)
	defer cancel()
	key := string(secret.Data["aws_access_key_id"])
	secretId := string(secret.Data["aws_secret_access_key"])
	region := string(secret.Data["region_name"])
	sessionToken := string(secret.Data["AWS_SESSION_TOKEN"])
	if err != nil {
		sessionToken = ""
	}

	cfg, err := awsconfig.LoadDefaultConfig(
		awsctx,
		awsconfig.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(key, secretId, sessionToken)),
		awsconfig.WithRegion(region),
	)

	checkErr(err)
	s3client := s3.NewFromConfig(cfg)

	job := spec.Jobs[0]
	loadETags := func(keys []string) []string {
		etags := []string{}
		if keys == nil {
			return etags
		}
		for _, prefix := range keys {
			worker := func(wg *sync.WaitGroup, page *s3.ListObjectsV2Output) {
				for _, obj := range page.Contents {
					if obj.Size > 0 {
						clean_etag := regexp.MustCompile(`[^a-zA-Z0-9 \s- ]+`).ReplaceAllString(*obj.ETag, "")
						etags = append(etags, clean_etag)
					}
				}
				wg.Done()
			}
			paginator := s3.NewListObjectsV2Paginator(s3client, &s3.ListObjectsV2Input{
				Bucket: &job.DataSource.Bucket,
				Prefix: &prefix,
			})

			var wg sync.WaitGroup
			for paginator.HasMorePages() {
				page, err := paginator.NextPage(awsctx)
				checkErr(err) // if bucket doesn't exist, this will raise error
				wg.Add(1)
				go worker(&wg, page)
			}
			wg.Wait()
		}
		return etags
	}

	train := job.DataSource.Keys.Train
	val := job.DataSource.Keys.Validation
	test := job.DataSource.Keys.Test
	etags := map[string]map[string][]string{
		"train": {
			"samples":   loadETags(train.Samples),
			"targets":   loadETags(train.Targets),
			"manifests": loadETags(train.Manifests),
		},
		"validation": {
			"samples":   loadETags(val.Samples),
			"targets":   loadETags(val.Targets),
			"manifests": loadETags(val.Manifests),
		},
		"test": {
			"samples":   loadETags(test.Samples),
			"targets":   loadETags(test.Targets),
			"manifests": loadETags(test.Manifests),
		},
	}

	nodeToExistEtagNum := map[string]int{}
	etagToNode := map[string]string{}

	// list existing etags on NFS
	for _, node := range nodes {
		addr := node.Status.Addresses[0].Address
		files, err := ioutil.ReadDir(fmt.Sprintf("/%s", addr))
		if err != nil {
			continue
		}
		for _, f := range files {
			etagToNode[f.Name()] = addr
		}
	}

	// find existing etags over all ETags
	var totalETags int
	var totalExistingEtags int
	for dataset := range etags {
		for data := range etags[dataset] {
			for _, etag := range etags[dataset][data] {
				if _, ok := etagToNode[etag]; ok {
					nodeToExistEtagNum[etagToNode[etag]] += 1
					totalExistingEtags += 1
				}
				totalETags += 1
			}
		}
	}

	// generate scheduling policy
	weights := map[string]float64{}
	allocatableResource := map[string]map[string]int64{}
	for _, node := range nodes {
		allocatable := node.Status.Allocatable
		gpus := allocatable["nvidia.com/gpu"]
		allocatableGPU, ok := gpus.AsInt64()
		if !ok {
			allocatableGPU = 0
		}
		allocatableResource[node.Status.Addresses[0].Address] = map[string]int64{
			"storage":        allocatable.StorageEphemeral().Value(),
			"nvidia.com/gpu": allocatableGPU,
		}
	}
	var totalCapacity float64 = 0
	for node := range allocatableResource {
		totalCapacity += float64(allocatableResource[node]["storage"])
	}

	/*
		We assign the job to the node that has maximum free space and sufficien GPU.
		Place dataset on other nodes sorted by free space.
	*/
	var requestGPU int64 = 0
	for _, job := range spec.Jobs {
		gpu := job.Resources.Requests["nvidia.com/gpu"]
		requestGPU += gpu.Value()
	}

	/*
		weights[node] = (allocatableGPU >= requestGPU) + (#existingETagsOnNode/#totalETags) + \
						(1-#totalExistingETags/#totalETags)*(allocatableNodeSpace/totalCapacity)
	*/
	for _, node := range nodes {
		addr := node.Status.Addresses[0].Address
		var left float64 = 0.0
		if val, ok := nodeToExistEtagNum[addr]; ok {
			left = float64(val) / float64(totalETags)
		}
		right := (1.0 - float64(totalExistingEtags)/float64(totalETags)) * float64(allocatableResource[addr]["storage"]) / totalCapacity
		weights[addr] = left + right
		if allocatableResource[addr]["nvidia.com/gpu"] >= requestGPU {
			weights[addr] += 1.0
		}
	}

	sort.SliceStable(nodeAddresses, func(i, j int) bool {
		return weights[nodeAddresses[i]] > weights[nodeAddresses[j]]
	})

	// generate schedule
	var schedule [][]string
	i := 0
	for ; i < len(nodeAddresses); i++ {
		address := nodeAddresses[i]
		hostname := addressHostMap[address]
		pair := []string{address, hostname}
		schedule = append(schedule, pair)
	}
	return schedule
}

func (r *DLCPodReconciler) createPod(ctx context.Context, dlcpod *v1alpha1.DLCPod) (corev1.Pod, error) {
	var pod corev1.Pod
	spec := dlcpod.Spec
	hostPathDir := corev1.HostPathDirectory
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
			Name: "shmem",
			VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{
				Path: "/dev/shm",
				Type: &hostPathDir,
			}},
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
	var jobNode corev1.Node
	nodes := r.filterNodes(ctx)
	// select node by user specified NodeSelector
	flag := false
	for key := range dlcpod.Spec.NodeSelector {
		for _, node := range nodes {
			if val, ok := node.Labels[key]; ok {
				if val == dlcpod.Spec.NodeSelector[key] {
					jobNode = node
					flag = true
					break
				}
			}
		}
		if flag {
			break
		}
	}

	// select node by user specified NodeName
	if !flag {
		for _, node := range nodes {
			if node.Status.Addresses[1].Address == dlcpod.Spec.NodeName {
				jobNode = node
				flag = true
				break
			}
		}
	}

	// select node based on scheduler decision
	if !flag {
		for _, node := range nodes {
			if node.Status.Addresses[0].Address == dlcpod.Spec.NodeSequence[0] {
				jobNode = node
				flag = true
				break
			}
		}
	}

	// move the jobNode to the head of NodeSequence
	if jobNode.Status.Addresses[0].Address != dlcpod.Spec.NodeSequence[0] {
		for i := 0; i < len(dlcpod.Spec.NodeSequence); i++ {
			if dlcpod.Spec.NodeSequence[i] == jobNode.Status.Addresses[0].Address {
				dlcpod.Spec.NodeSequence = append(dlcpod.Spec.NodeSequence[:i], dlcpod.Spec.NodeSequence[i+1:]...)
				break
			}
		}
		dlcpod.Spec.NodeSequence = append(dlcpod.Spec.NodeSequence[:1], dlcpod.Spec.NodeSequence[0:]...)
		dlcpod.Spec.NodeSequence[0] = jobNode.Status.Addresses[0].Address
	}

	for _, node := range nodes {
		nodeip := node.Status.Addresses[0].Address

		// for local NFS server, we directly mount to the NFS mount point
		hostPathDirectoryOrCreate := corev1.HostPathDirectoryOrCreate
		if nodeip == jobNode.Status.Addresses[0].Address {
			volumes = append(volumes, corev1.Volume{
				Name: strings.ReplaceAll(nodeip, ".", "-"),
				VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{
					Path: "/nfs_storage",
					Type: &hostPathDirectoryOrCreate,
				}},
			})
		} else {
			volumes = append(volumes, corev1.Volume{
				Name: strings.ReplaceAll(nodeip, ".", "-"),
				VolumeSource: corev1.VolumeSource{NFS: &corev1.NFSVolumeSource{
					Server:   nodeip,
					Path:     "/nfs_storage",
					ReadOnly: false,
				}},
			})
		}

		vol_mounts = append(vol_mounts, corev1.VolumeMount{
			Name:      strings.ReplaceAll(nodeip, ".", "-"),
			MountPath: fmt.Sprintf("/%s", nodeip),
		})
	}

	alloc := jobNode.Status.Allocatable
	gpus := alloc["nvidia.com/gpu"]
	allocGPU, ok := gpus.AsInt64()
	if !ok || allocGPU == 0 {
		allocGPU = 1
	}
	var containers []corev1.Container
	// alph := 0.8
	for _, job := range spec.Jobs {
		env := job.Env
		env = append(env, corev1.EnvVar{Name: "JOBNAME", Value: job.Name})
		// cpuLimit := int64(alph * float64(alloc.Cpu().MilliValue()/allocGPU/int64(len(spec.Jobs))))
		// memoryLimit := int64(alph * float64(alloc.Memory().MilliValue()/allocGPU/int64(len(spec.Jobs))))
		// ephemeralStorageLimit := int64(alph * float64(alloc.StorageEphemeral().MilliValue()/allocGPU/int64(len(spec.Jobs))))
		// gpuLimit := job.Resources.Limits["nvidia.com/gpu"]
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
			// Resources: corev1.ResourceRequirements{
			// 	Limits: corev1.ResourceList{
			// 		"cpu":               *resource.NewMilliQuantity(cpuLimit, resource.DecimalSI),
			// 		"memory":            *resource.NewMilliQuantity(memoryLimit, resource.DecimalSI),
			// 		"ephemeral-storage": *resource.NewMilliQuantity(ephemeralStorageLimit, resource.DecimalSI),
			// 		// "nvidia.com/gpu":    *resource.NewQuantity(gpuLimit.Value(), resource.DecimalSI),
			// 	},
			// 	Requests: job.Resources.Requests,
			// },
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
			RestartPolicy: corev1.RestartPolicyAlways,
			NodeName:      jobNode.Name,
			HostNetwork:   spec.HostNetwork,
		},
	}

	err := ctrl.SetControllerReference(dlcpod, &pod, r.Scheme)
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
			"name":         job.Name,
			"datasource":   job.DataSource,
			"nodesequence": dlcpod.Spec.NodeSequence,
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
