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
sudo echo "/nfs_storage $cidr(rw,sync,no_all_squash,no_subtree_check)" >> /etc/exports
sudo exportfs -rv
sudo systemctl restart nfs-kernel-server


# sudo mkdir /129.59.234.238
# sudo mkdir /129.59.234.236
# sudo mount -t nfs 129.59.234.238:/nfs_storage /129.59.234.238
# sudo mount -t nfs 129.59.234.236:/nfs_storage /129.59.234.236

# sudo mkdir /129.59.234.240
# sudo mkdir /129.59.234.236
# sudo mount -t nfs 129.59.234.240:/nfs_storage /129.59.234.240
# sudo mount -t nfs 129.59.234.236:/nfs_storage /129.59.234.236

# sudo mkdir /129.59.234.240
# sudo mkdir /129.59.234.238
# sudo mount -t nfs 129.59.234.240:/nfs_storage /129.59.234.240
# sudo mount -t nfs 129.59.234.238:/nfs_storage /129.59.234.238