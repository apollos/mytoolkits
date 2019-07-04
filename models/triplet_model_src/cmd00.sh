source /opt/DL/tensorflow/bin/tensorflow-activate
source /opt/DL/caffe/bin/caffe-activate
source /opt/DL/nccl/bin/nccl-activate


export SIG=/var/tmp/dlisig/
export SHARED=/shared/dli_shared_fs/aienterpriseR5
export PYTHONPATH=/opt/DL/tensorboard/lib/python2.7/site-packages/:$SIG/spark-1.6.1-hadoop-2.6/python/lib/pyspark.zip$SIG/spark-1.6.1-hadoop-2.6/python/lib/py4j-0.9-src.zip:$SIG/spark-1.6.1-hadoop-2.6/lib/ego/spark-core_2.10-1.6.1.jar:$SHARED/tools/parameter_mgr:$SHARED/fabric/tools:$SHARED/models/TensorFlow/f-frcnn-2/f-frcnn-2-209500-fabriconly:$SIG/spark-1.6.1-hadoop-2.6/python/:$PYTHONPATH
#export EGO_TOP=/opt/ibm/spectrum_mpi
#export EGO_TOP=/home/steven/sc/
export EGO_TOP=/home/bavdicev/sc/
port1=`shuf -i 40000-60000 -n 1`
port1a=`shuf -i 40000-60000 -n 1`
port2=`shuf -i 40000-60000 -n 1`
port3=`shuf -i 40000-60000 -n 1`
port4=`shuf -i 40000-60000 -n 1`
port5=`shuf -i 40000-60000 -n 1`
export CUDA_VISIBLE_DEVICES=
python main_ocr.py --ps_hosts=localhost:$port1,localhost:$port1a --worker_hosts=localhost:$port2,localhost:$port3,localhost:$port4,localhost:$port5 --task_id=0 --job_name=ps --train_dir=train/ &
pspid1=$!
export CUDA_VISIBLE_DEVICES=
python main_ocr.py --ps_hosts=localhost:$port1,localhost:$port1a --worker_hosts=localhost:$port2,localhost:$port3,localhost:$port4,localhost:$port5 --task_id=1 --job_name=ps --train_dir=train/ &
pspid1a=$!
export CUDA_VISIBLE_DEVICES=0
python main_ocr.py --ps_hosts=localhost:$port1 --worker_hosts=localhost:$port2,localhost:$port3,localhost:$port4,localhost:$port5 --task_id=0 --job_name=worker --train_dir=train/ &
export CUDA_VISIBLE_DEVICES=1
python main_ocr.py --ps_hosts=localhost:$port1 --worker_hosts=localhost:$port2,localhost:$port3,localhost:$port4,localhost:$port5 --task_id=1 --job_name=worker --train_dir=train/ &
export CUDA_VISIBLE_DEVICES=2
python main_ocr.py --ps_hosts=localhost:$port1 --worker_hosts=localhost:$port2,localhost:$port3,localhost:$port4,localhost:$port5 --task_id=2 --job_name=worker --train_dir=train/ &
export CUDA_VISIBLE_DEVICES=3
python main_ocr.py --ps_hosts=localhost:$port1 --worker_hosts=localhost:$port2,localhost:$port3,localhost:$port4,localhost:$port5 --task_id=3 --job_name=worker --train_dir=train/
kill -9 $pspid1
kill -9 $pspid1a
