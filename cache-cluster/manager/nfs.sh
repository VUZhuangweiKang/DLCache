#!/bin/bash
if [ $# -eq 0 ]
  then
    cidr="129.59.234.0/24"
else
    cidr=$1
fi
# execute this script on all worker nodes
sudo apt install -y nfs-kernel-server
sudo mkdir /nfs_storage
sudo chmod -R 777 /nfs_storage/
sudo echo "/nfs_storage $cidr(insecure,rw,sync,no_root_squash)" >> /etc/exports
sudo exportfs -rv
sudo systemctl restart nfs-kernel-server


machines=("129.59.234.236" "129.59.234.237" "129.59.234.238" "129.59.234.239" "129.59.234.240" "129.59.234.241")
for mach in "${machines[@]}"; do
    sudo mkdir /$mach
    sudo mount -t nfs $mach:/nfs_storage /$mach
done