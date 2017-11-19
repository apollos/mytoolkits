source /opt/DL/tensorflow/bin/tensorflow-activate
source /opt/DL/caffe/bin/caffe-activate
source /opt/DL/nccl/bin/nccl-activate


export PYTHONPATH=/home/egoadmin/gpusigtest/spark-1.6.1-hadoop-2.6/python/lib/pyspark.zip:/home/egoadmin/gpusigtest/spark-1.6.1-hadoop-2.6/python/lib/py4j-0.9-src.zip:/home/egoadmin/gpusigtest/spark-1.6.1-hadoop-2.6/lib/ego/spark-core_2.10-1.6.1.jar:/mygpfs/tools/parameter_mgr:/mygpfs/fabric/tools:/mygpfs/models/TensorFlow/f-frcnn-2/f-frcnn-2-209500-fabriconly:/home/egoadmin/gpusigtest/spark-1.6.1-hadoop-2.6/python/lib/pyspark.zip:/home/egoadmin/gpusigtest/spark-1.6.1-hadoop-2.6/python/:$PYTHONPATH
export EGO_TOP=/opt/ibm/spectrumcomputing
export CUDA_VISIBLE_DEVICES=
python main.py --ps_hosts=cit1075.rtp.raleigh.ibm.com:51922 --worker_hosts=cit1075.rtp.raleigh.ibm.com:52335 --task_id=0 --job_name=ps --train_dir=train/ &
pspid=$!
export CUDA_VISIBLE_DEVICES=1
python main.py --ps_hosts=cit1075.rtp.raleigh.ibm.com:51922 --worker_hosts=cit1075.rtp.raleigh.ibm.com:52335 --task_id=0 --job_name=worker --train_dir=train/
kill -9 $pspid