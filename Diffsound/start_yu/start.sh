#python /apdcephfs/share_1316500/donchaoyang/code3/DiffusionFast/running_command/run_train_vgg.py
python /apdcephfs/share_1316500/donchaoyang/code3/DiffusionFast/running_command/run_train_caps_pre_vgg.py
#python /apdcephfs/share_1316500/donchaoyang/code3/VQ-Diffusion/running_command/run_train_caps.py
#python /apdcephfs/share_1316500/donchaoyang/code3/VQ-Diffusion/running_command/run_train_caps_2048.py
#python /apdcephfs/share_1316500/donchaoyang/code3/VQ-Diffusion/running_command/run_train_caps_512.py
#python /apdcephfs/share_1316500/donchaoyang/code3/VQ-Diffusion/running_command/run_train_caps_small.py
#python /apdcephfs/share_1316500/donchaoyang/code3/DiffusionFast/running_command/run_train_audioset.py
# flag=multi_gpu
# echo chief ip:  $CHIEF_IP
# host_ip_list=(${NODE_IP_LIST//,/ })
# let world_size=NODE_NUM/8
# echo world size: $world_size
# rank=1
# for host in ${host_ip_list[@]}; do
#     echo $host
#     host_ip=$(echo $host| cut -d':' -f 1)
#     echo $host_ip
#     if [ $host_ip != $CHIEF_IP ]; then
#         ssh -o StrictHostKeyChecking=no -t -f $host_ip "cd /apdcephfs/share_1316500/donchaoyang/code3/VQ-Diffusion/start; pwd; sh train.sh  $world_size $rank $CHIEF_IP"
#         let rank+=1
#         echo rank: $rank
#     fi
# done
# echo start chief node
# sh train.sh  $world_size 0 $CHIEF_IP
# export LD_LIBRARY_PATH=/apdcephfs/private_donchaoyang/anaconda3/lib:$LD_LIBRARY_PATH
# export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=info
# master_port=22275
# MASTER_PORT=$master_port
# echo nproc_per_node: ${HOST_GPU_NUM}
# echo nnodes: ${HOST_NUM}
# echo node_rank: ${INDEX}
# echo master_addr: ${CHIEF_IP}
# NCCL_DEBUG=TRACE /apdcephfs/private_donchaoyang/anaconda3/bin/python -m torch.distributed.launch \
#     --nproc_per_node ${HOST_GPU_NUM} --master_port $MASTER_PORT \
#     --nnodes=${HOST_NUM} --node_rank=${INDEX} --master_addr=${CHIEF_IP} \
#     /apdcephfs/share_1316500/donchaoyang/code3/M-Diffusion/train_spec_m.py \
#     --name audioset_train \
#     --config_file /apdcephfs/share_1316500/donchaoyang/code3/VQ-Diffusion/configs/audioset.yaml \
#     --tensorboard 
