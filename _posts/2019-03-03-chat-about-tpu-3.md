---
title: "闲话TPU #3 模型编写"
categories: "TPU Blog"
author: "Cy.Feng"
date: 2019-03-05
---

版权声明：本博文欢迎分享与转载，转载请注明出处和作者。<cy.z.feng@gmail.com>

> [闲话TPU #1 背景/价格/TFRC计划及羊毛](http://cyfeng.science/tpu/blog/2019/03/07/chat-about-tpu-1.html)
>
> [闲话TPU #2 配置GCP环境/创建TPU实例](http://cyfeng.science/tpu/blog/2019/03/06/chat-about-tpu-2.html)
>
> [闲话TPU #4 Coral Edge TPU赋能移动端](http://cyfeng.science/tpu/blog/2019/03/04/chat-about-tpu-4.html)

## 叁/编写适用于TPU的模型

## Three/Coding

从入了DL的坑之后, up也算使用过不少的framework. 从最初在自己的机器上跑caffe到使用Keras, 从臃肿的TensorFlow转向当时正火爆的动态图多GPU数据并行PyTorch, 后来经历PyTorch和Caffe2合并从0.4.0望穿秋水的等这1.0的跨越的时候, 被借助TPU/FPGA等超级加速硬件怼paper到哭, ~~打不过就加入~~下定决心也要借助计算红利快速迭代创造一波价值. 

[后说后话, ~~因为马上要拥有DGX-2了嘿嘿,~~ 现在又在逐步回归Pytorch-1.0.1]渐渐也就理解了个事er, **idea跟人走, framework跟硬件走**, 不敢说走着走着路更宽了, 逐渐进步还是有的. ^ ^

回归正题, 如上所述TPU严格要求所对应的TF版本. 现可用版本如下表所示:

| TensorFlow version | Cloud TPU support start | Cloud TPU support end  |
| ------------------ | ----------------------- | ---------------------- |
| 1.13               | March 11, 2019          | (End date not yet set) |
| 1.12               | November 8, 2018        | (End date not yet set) |
| 1.11               | September 27, 2018      | (End date not yet set) |
| ~~1.9~~            | ~~July 2, 2018~~        | ~~March 11, 2019~~     |
| ~~1.8~~            | ~~April 20, 2018~~      | ~~March 11, 2019~~     |

建议的模型书写模式(up用的比较熟练的一个模式):

`tf.data`用来进行data ingestion and transformation via parallel input-pipeline.

`tf.estimator`用来build your models.

`Eager execution`模式训练我们的模型

> 为了最大获取TPU的加速比, Shapes应在模型运行时就是明确的, 因此要掌握dynamic的尺度.
>
> The XLA compiler compiles a TensorFlow graph just in time for the first batch. If any subsequent batches have different shapes, the model doesn't work. (Re-compiling the graph every time the shape changes is too slow.) Therefore, any model that has tensors with dynamic shapes that change at runtime isn’t well suited to TPUs.

`TensorFlow Serving`进行灵活的高性能服务部署.

> RESTful API: https://www.tensorflow.org/tfx/serving/api_rest

具体的模型编写框架可以参照tensorflow/tpu在GitHub上的[repo](https://github.com/tensorflow/tpu), 可以大致如下形容:

`preprocessing.py`

```python
def	preprocess_ops(xxx):
    pass
def preprocess_for_train(xxx):
    pass
def preprocess_for_eval(xxx):
    pass
```

`model.py`

```python
	def model(xxx):
        if model is tf.keras based:
            # start with a Input layer and end with layer-like function
            return model
        else:
            return model.output
```

`params.py`

```python
defaults_params = dict(
	...
)
```

`inputpipeline.py`

```python
import preprocessing

class InputPipeline(xxx):
	do some preprocessing
    return tf.dataset(ooo)
```

`main.py`

```python
import inputpipeline
import params
import model

	# some FLAGS used when run shell

def model_fn(features, labels, mode, params):
    # def network from model.py
    return TPUEstimatorSpec(when train|eval|predict)

def main():
    # tf.contrib.cluster_resolver.TPUClusterResolver
    tpu_cluster_resolver(tpu_gRPC_name, zone, project)
    # tf.contrib.tpu.RunConfig
    config(cluster, model_dir, session_config, tpu_config)
    # tf.contrib.tpu.TPUEstimator
    classifier = TPUEstimator(use_tpu, model_fn, config, params, batch_size...)
    train_input, eval_input = inputpipeline(is_training, data_dir, num_paralell_calls, use_bfloat16)
    
    if FLAGS.mode == EVAL:
        pass
    elif FlAGS.mode == TRAIN|TRAIN_AND_EVAL:
        pass

if __name__ == '__main__':
    app.run(main)
    
```

Coding这方面其实没有多大的变化, 将estimator用TPUestimator封装, TensorBoard inside, 参考着offical [repo](https://github.com/tensorflow/tpu) 和我们常规使用没有太大差别.

**然后请务必多次运行并尝试!Try/Try/Try!**





## 肆/错误排除

## Four/Troubleshooting

Wrong Message:

```markdown
WARNING:tensorflow:Estimator's model_fn (<function resnet_model_fn at 0x7f44d31fd730>) includes params argument, but params are not passed to Estimator.
INFO:tensorflow:Using config: {'_save_checkpoints_steps': 1251, '_evaluation_master': 'grpc://10.2.3.2:8470', '_session_config': graph_options {
  rewrite_options {
    disable_meta_optimizer: true
  }
}
cluster_def {
  job {
    name: "worker"
    tasks {
      value: "10.2.3.2:8470"
    }
  }
}
, '_log_step_count_steps': None, '_keep_checkpoint_max': 8, '_task_id': 0, '_global_id_in_cluster': 0, '_eval_distribute': None, '_protocol': None, '_master': 'grpc://10.2.3.2:8470', '_experimental_distribute': None, '_tpu_config': TPUConfig(iterations_per_loop=1251, num_shards=8, num_cores_per_replica=None, per_host_input_for_training=3, tpu_job_name=None, initial_infeed_sleep_secs=None, input_partition_dims=None), '_is_chief': True, '_keep_checkpoint_every_n_hours': 10000, '_save_checkpoints_secs': None, '_model_dir': 'gs://my_results/resnet-tpu-framework/weighted-resnet-4', '_save_summary_steps': 100, '_train_distribute': None, '_task_type': 'worker', '_num_worker_replicas': 1, '_service': None, '_device_fn': None, '_num_ps_replicas': 0, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7f44d2fd3d68>, '_tf_random_seed': 16, '_cluster': <tensorflow.contrib.cluster_resolver.python.training.tpu_cluster_resolver.TPUClusterResolver object at 0x7f44d3202748>}
INFO:tensorflow:_TPUContext: eval_on_tpu True
INFO:tensorflow:Precision: bfloat16
INFO:tensorflow:Using dataset: gs://my_datasets/data
INFO:tensorflow:Waiting for new checkpoint at gs://my_results/resnet-tpu-framework/weighted-resnet-4
terminate called after throwing an instance of 'std::bad_alloc'
  what():  std::bad_alloc
Aborted (core dumped)
```

因为VM同时后台运行多个任务(nohup)导致内存不足, 增大VM内存即可

Multiple missions run in background(within nohup mode) causing the lack of memory of VM, just increase the memory in VM specification.





```markdown
INFO:tensorflow:Error recorded from outfeed: All 10 retry attempts failed. The last failure: Unavailable: Error executing an HTTP request: libcurl code 28 meaning 'Timeout was reached', error details: SSL connection timeout
	 when initiating an upload to gs://my_results/resnet-tpu-framework/weighted-resnet-3/events.out.tfevents.1546877726.n-ec458f18-w-0.v2
	Failed to sync 715 events to gs://my_results/resnet-tpu-framework/weighted-resnet-3/events.out.tfevents.1546877726.n-ec458f18-w-0.v2
	Could not flush events file.
	 [[node current_epoch (defined at ./resnet_main.py:393)  = WriteScalarSummary[T=DT_FLOAT, _device="/job:worker/replica:0/task:0/device:CPU:0"](SummaryWriter, strided_slice, current_epoch/tag, current_epoch/Identity)]]

Caused by op 'current_epoch', defined at:
  File "./resnet_main.py", line 577, in <module>
    tf.app.run()
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/platform/app.py", line 125, in run
    _sys.exit(main(argv))
  File "./resnet_main.py", line 564, in main
    hooks=hooks)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/contrib/tpu/python/tpu/tpu_estimator.py", line 2403, in train
    saving_listeners=saving_listeners
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/estimator/estimator.py", line 354, in train
    loss = self._train_model(input_fn, hooks, saving_listeners)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/estimator/estimator.py", line 1207, in _train_model
    return self._train_model_default(input_fn, hooks, saving_listeners)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/estimator/estimator.py", line 1237, in _train_model_default
    features, labels, model_fn_lib.ModeKeys.TRAIN, self.config)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/contrib/tpu/python/tpu/tpu_estimator.py", line 2195, in _call_model_fn
    features, labels, mode, config)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/estimator/estimator.py", line 1195, in _call_model_fn
    model_fn_results = self._model_fn(features=features, **kwargs)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/contrib/tpu/python/tpu/tpu_estimator.py", line 2503, in _model_fn
    host_ops = host_call.create_tpu_hostcall()
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/contrib/tpu/python/tpu/tpu_estimator.py", line 1736, in create_tpu_hostcall
    ret[name] = self._host_fns[name](*dequeue_ops)
  File "./resnet_main.py", line 393, in host_call_fn
    summary.scalar('current_epoch', ce[0], step=gs)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/ops/summary_ops_v2.py", line 440, in scalar
    return summary_writer_function(name, tensor, function, family=family)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/ops/summary_ops_v2.py", line 384, in summary_writer_function
    should_record_summaries(), record, _nothing, name="")
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/framework/smart_cond.py", line 54, in smart_cond
    return true_fn()
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/ops/summary_ops_v2.py", line 377, in record
    with ops.control_dependencies([function(tag, scope)]):
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/ops/summary_ops_v2.py", line 438, in function
    name=scope)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/ops/gen_summary_ops.py", line 633, in write_scalar_summary
    name=name)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/framework/op_def_library.py", line 787, in _apply_op_helper
    op_def=op_def)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/util/deprecation.py", line 488, in new_func
    return func(*args, **kwargs)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 3274, in create_op
    op_def=op_def)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 1770, in __init__
    self._traceback = tf_stack.extract_stack()

AbortedError (see above for traceback): All 10 retry attempts failed. The last failure: Unavailable: Error executing an HTTP request: libcurl code 28 meaning 'Timeout was reached', error details: SSL connection timeout
	 when initiating an upload to gs://my_results/resnet-tpu-framework/weighted-resnet-3/events.out.tfevents.1546877726.n-ec458f18-w-0.v2
	Failed to sync 715 events to gs://my_results/resnet-tpu-framework/weighted-resnet-3/events.out.tfevents.1546877726.n-ec458f18-w-0.v2
	Could not flush events file.
	 [[node current_epoch (defined at ./resnet_main.py:393)  = WriteScalarSummary[T=DT_FLOAT, _device="/job:worker/replica:0/task:0/device:CPU:0"](SummaryWriter, strided_slice, current_epoch/tag, current_epoch/Identity)]]
```

疑似是VM和GCS之间网络连接的问题

Bad internet access. Change and try again will solve the problem.



```markdown
FailedPreconditionError (see above for traceback): Unable to enqueue when not opened, queue: [0000:00:05.0 PE0 C1 MC2 TN0 Queue TENSOR_CORE_INFEED]. State is: FAILED
	 [[node input_pipeline_task0/while/InfeedQueue/enqueue/2 (defined at /home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/contrib/tpu/ops/gen_tpu_ops.py:1055)  = InfeedEnqueueTuple[_class=["loc:@input_pipeline_task0/while/IteratorGetNext_2"], device_ordinal=2, dtypes=[DT_BFLOAT16, DT_INT32], shapes=[[19267584], [128]], _device="/job:worker/replica:0/task:0/device:CPU:0"](input_pipeline_task0/while/IteratorGetNext_2, input_pipeline_task0/while/IteratorGetNext_2:1)]]

INFO:tensorflow:An error was raised. This may be due to a preemption in a connected worker or parameter server. The current session will be closed and a new session will be created. This error may also occur due to a gRPC failure caused by high memory or network bandwidth usage in the parameter servers. If this error occurs repeatedly, try increasing the number of parameter servers assigned to the job. Error: Socket closed
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Restoring parameters from gs://my_results/resnet-tpu-framework/weighted-resnet-9/model.ckpt-0
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 0 into gs://my_results/resnet-tpu-framework/weighted-resnet-9/model.ckpt.
INFO:tensorflow:Initialized dataset iterators in 0 seconds
INFO:tensorflow:Installing graceful shutdown hook.
2019-01-10 15:02:51.275101: W tensorflow/core/distributed_runtime/rpc/grpc_session.cc:349] GrpcSession::ListDevices will initialize the session with an empty graph and other defaults because the session has not yet been created.
INFO:tensorflow:Creating heartbeat manager for ['/job:tpu_worker/replica:0/task:0/device:CPU:0']
INFO:tensorflow:Configuring worker heartbeat: shutdown_mode: WAIT_FOR_COORDINATOR

INFO:tensorflow:Init TPU system
INFO:tensorflow:Initialized TPU in 6 seconds
INFO:tensorflow:Starting infeed thread controller.
INFO:tensorflow:Starting outfeed thread controller.
INFO:tensorflow:Enqueue next (1251) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (1251) batch(es) of data from outfeed.

```

生成TPU的时候状态不正确? 等待了一会之后出现了WAIT_FOR_COORDINATOR之后成功初始化TPU.

TPU status is wrong(not sure why :( ). Just waiting for another minute, a massage ‘WAIT_FOR_COORDINATOR’ come out and the TPU is successfully initialized.



```markdown
INFO:tensorflow:Querying Tensorflow master (grpc://10.3.2.2:8470) for TPU system metadata.
2019-01-13 03:15:46.904944: W tensorflow/core/distributed_runtime/rpc/grpc_session.cc:349] GrpcSession::ListDevices will initialize the session with an empty graph and other defaults because the session has not yet been created.
WARNING:tensorflow:Failed to connect to the Tensorflow master. The TPU worker may not be ready (still scheduling) or the Tensorflow master address is incorrect: got (grpc://10.3.2.2:8470).
WARNING:tensorflow:Retrying (10/120).
INFO:tensorflow:Querying Tensorflow master (grpc://10.3.2.2:8470) for TPU system metadata.
2019-01-13 03:16:46.908810: W tensorflow/core/distributed_runtime/rpc/grpc_session.cc:349] GrpcSession::ListDevices will initialize the session with an empty graph and other defaults because the session has not yet been created.
WARNING:tensorflow:Failed to connect to the Tensorflow master. The TPU worker may not be ready (still scheduling) or the Tensorflow master address is incorrect: got (grpc://10.3.2.2:8470).
WARNING:tensorflow:Retrying (11/120).
INFO:tensorflow:Querying Tensorflow master (grpc://10.3.2.2:8470) for TPU system metadata.
2019-01-13 03:17:46.913479: W tensorflow/core/distributed_runtime/rpc/grpc_session.cc:349] GrpcSession::ListDevices will initialize the session with an empty graph and other defaults because the session has not yet been created.
WARNING:tensorflow:Failed to connect to the Tensorflow master. The TPU worker may not be ready (still scheduling) or the Tensorflow master address is incorrect: got (grpc://10.3.2.2:8470).
WARNING:tensorflow:Retrying (12/120).

```

gRPC连接问题.

Connection problem with gRPC. Reset the TPU, or just init another TPU. All of those kinds of problems can be solved through restarting and recreating in high probability.



```markdown
INFO:tensorflow:Init TPU system
INFO:tensorflow:Initialized TPU in 2 seconds
INFO:tensorflow:Starting infeed thread controller.
INFO:tensorflow:Starting outfeed thread controller.
INFO:tensorflow:Enqueue next (1251) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (1251) batch(es) of data from outfeed.
INFO:tensorflow:Error recorded from outfeed: Step was cancelled by an explicit call to `Session::Close()`.
INFO:tensorflow:Error recorded from training_loop: Compilation failure: Ran out of memory in memory space vmem. It should not be possible to run out of vmem - please file a bug against XLA.

Largest program allocations in vmem:

  XLA label: register allocator spill slots
  Allocation type: scoped

  XLA label: %fusion.177 = (bf16[], f32[256]{0}, f32[256]{0}, bf16[128,56,56,256]{3,0,2,1}) fusion(f32[256]{0}, f32[256]{0}, f32[256]{0}, f32[256]{0}, ...(+13)), kind=kOutput, calls=%fused_computation.177, sharding={ {maximal device=0}, {maximal device=0}, {maximal devi...} }
  Allocation type: scoped

  XLA label: %fusion.177 = (bf16[], f32[256]{0}, f32[256]{0}, bf16[128,56,56,256]{3,0,2,1}) fusion(f32[256]{0}, f32[256]{0}, f32[256]{0}, f32[256]{0}, ...(+13)), kind=kOutput, calls=%fused_computation.177, sharding={ {maximal device=0}, {maximal device=0}, {maximal devi...} }
  Allocation type: scoped

  XLA label: %fusion.177 = (bf16[], f32[256]{0}, f32[256]{0}, bf16[128,56,56,256]{3,0,2,1}) fusion(f32[256]{0}, f32[256]{0}, f32[256]{0}, f32[256]{0}, ...(+13)), kind=kOutput, calls=%fused_computation.177, sharding={ {maximal device=0}, {maximal device=0}, {maximal devi...} }
  Allocation type: scoped

  XLA label: %fusion.177 = (bf16[], f32[256]{0}, f32[256]{0}, bf16[128,56,56,256]{3,0,2,1}) fusion(f32[256]{0}, f32[256]{0}, f32[256]{0}, f32[256]{0}, ...(+13)), kind=kOutput, calls=%fused_computation.177, sharding={ {maximal device=0}, {maximal device=0}, {maximal devi...} }
  Allocation type: scoped

	TPU compilation failed
	 [[{{node tpu_compile_succeeded_assert/_2211098028679383727/_435}} = TPUCompileSucceededAssert[_device="/job:worker/replica:0/task:0/device:CPU:0"](TPUReplicate/_compile/_11277858145685444465/_434)]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.

	 [[{{node TPUReplicate/_compile/_11277858145685444465/_434/after_compilation/_436_G6101}} = _Recv[client_terminated=false, recv_device="/job:worker/replica:0/task:0/device:TPU:6", send_device="/job:worker/replica:0/task:0/device:CPU:0", send_device_incarnation=452505474266104386, tensor_name="edge_6943_...ation/_436", tensor_type=DT_FLOAT, _device="/job:worker/replica:0/task:0/device:TPU:6"]()]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.

INFO:tensorflow:training_loop marked as finished
WARNING:tensorflow:Reraising captured error
Traceback (most recent call last):
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 1334, in _do_call
    return fn(*args)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 1319, in _run_fn
    options, feed_dict, fetch_list, target_list, run_metadata)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 1407, in _call_tf_sessionrun
    run_metadata)
tensorflow.python.framework.errors_impl.ResourceExhaustedError: Compilation failure: Ran out of memory in memory space vmem. It should not be possible to run out of vmem - please file a bug against XLA.

Largest program allocations in vmem:

  XLA label: register allocator spill slots
  Allocation type: scoped

  XLA label: %fusion.177 = (bf16[], f32[256]{0}, f32[256]{0}, bf16[128,56,56,256]{3,0,2,1}) fusion(f32[256]{0}, f32[256]{0}, f32[256]{0}, f32[256]{0}, ...(+13)), kind=kOutput, calls=%fused_computation.177, sharding={ {maximal device=0}, {maximal device=0}, {maximal devi...} }
  Allocation type: scoped

  XLA label: %fusion.177 = (bf16[], f32[256]{0}, f32[256]{0}, bf16[128,56,56,256]{3,0,2,1}) fusion(f32[256]{0}, f32[256]{0}, f32[256]{0}, f32[256]{0}, ...(+13)), kind=kOutput, calls=%fused_computation.177, sharding={ {maximal device=0}, {maximal device=0}, {maximal devi...} }
  Allocation type: scoped

  XLA label: %fusion.177 = (bf16[], f32[256]{0}, f32[256]{0}, bf16[128,56,56,256]{3,0,2,1}) fusion(f32[256]{0}, f32[256]{0}, f32[256]{0}, f32[256]{0}, ...(+13)), kind=kOutput, calls=%fused_computation.177, sharding={ {maximal device=0}, {maximal device=0}, {maximal devi...} }
  Allocation type: scoped

  XLA label: %fusion.177 = (bf16[], f32[256]{0}, f32[256]{0}, bf16[128,56,56,256]{3,0,2,1}) fusion(f32[256]{0}, f32[256]{0}, f32[256]{0}, f32[256]{0}, ...(+13)), kind=kOutput, calls=%fused_computation.177, sharding={ {maximal device=0}, {maximal device=0}, {maximal devi...} }
  Allocation type: scoped

	TPU compilation failed
	 [[{{node tpu_compile_succeeded_assert/_2211098028679383727/_435}} = TPUCompileSucceededAssert[_device="/job:worker/replica:0/task:0/device:CPU:0"](TPUReplicate/_compile/_11277858145685444465/_434)]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.

	 [[{{node TPUReplicate/_compile/_11277858145685444465/_434/after_compilation/_436_G6101}} = _Recv[client_terminated=false, recv_device="/job:worker/replica:0/task:0/device:TPU:6", send_device="/job:worker/replica:0/task:0/device:CPU:0", send_device_incarnation=452505474266104386, tensor_name="edge_6943_...ation/_436", tensor_type=DT_FLOAT, _device="/job:worker/replica:0/task:0/device:TPU:6"]()]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "./resnet_main.py", line 585, in <module>
    tf.app.run()
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/platform/app.py", line 125, in run
    _sys.exit(main(argv))
  File "./resnet_main.py", line 572, in main
    hooks=hooks)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/contrib/tpu/python/tpu/tpu_estimator.py", line 2409, in train
    rendezvous.raise_errors()
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/contrib/tpu/python/tpu/error_handling.py", line 128, in raise_errors
    six.reraise(typ, value, traceback)
  File "/usr/lib/python3/dist-packages/six.py", line 686, in reraise
    raise value
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/contrib/tpu/python/tpu/tpu_estimator.py", line 2403, in train
    saving_listeners=saving_listeners
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/estimator/estimator.py", line 354, in train
    loss = self._train_model(input_fn, hooks, saving_listeners)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/estimator/estimator.py", line 1207, in _train_model
    return self._train_model_default(input_fn, hooks, saving_listeners)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/estimator/estimator.py", line 1241, in _train_model_default
    saving_listeners)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/estimator/estimator.py", line 1471, in _train_with_estimator_spec
    _, loss = mon_sess.run([estimator_spec.train_op, estimator_spec.loss])
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/training/monitored_session.py", line 671, in run
    run_metadata=run_metadata)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/training/monitored_session.py", line 1156, in run
    run_metadata=run_metadata)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/training/monitored_session.py", line 1255, in run
    raise six.reraise(*original_exc_info)
  File "/usr/lib/python3/dist-packages/six.py", line 686, in reraise
    raise value
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/training/monitored_session.py", line 1240, in run
    return self._sess.run(*args, **kwargs)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/training/monitored_session.py", line 1312, in run
    run_metadata=run_metadata)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/training/monitored_session.py", line 1076, in run
    return self._sess.run(*args, **kwargs)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 929, in run
    run_metadata_ptr)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 1152, in _run
    feed_dict_tensor, options, run_metadata)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 1328, in _do_run
    run_metadata)
  File "/home/cyfeng16/.local/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 1348, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.ResourceExhaustedError: Compilation failure: Ran out of memory in memory space vmem. It should not be possible to run out of vmem - please file a bug against XLA.

Largest program allocations in vmem:

  XLA label: register allocator spill slots
  Allocation type: scoped

  XLA label: %fusion.177 = (bf16[], f32[256]{0}, f32[256]{0}, bf16[128,56,56,256]{3,0,2,1}) fusion(f32[256]{0}, f32[256]{0}, f32[256]{0}, f32[256]{0}, ...(+13)), kind=kOutput, calls=%fused_computation.177, sharding={ {maximal device=0}, {maximal device=0}, {maximal devi...} }
  Allocation type: scoped

  XLA label: %fusion.177 = (bf16[], f32[256]{0}, f32[256]{0}, bf16[128,56,56,256]{3,0,2,1}) fusion(f32[256]{0}, f32[256]{0}, f32[256]{0}, f32[256]{0}, ...(+13)), kind=kOutput, calls=%fused_computation.177, sharding={ {maximal device=0}, {maximal device=0}, {maximal devi...} }
  Allocation type: scoped

  XLA label: %fusion.177 = (bf16[], f32[256]{0}, f32[256]{0}, bf16[128,56,56,256]{3,0,2,1}) fusion(f32[256]{0}, f32[256]{0}, f32[256]{0}, f32[256]{0}, ...(+13)), kind=kOutput, calls=%fused_computation.177, sharding={ {maximal device=0}, {maximal device=0}, {maximal devi...} }
  Allocation type: scoped

  XLA label: %fusion.177 = (bf16[], f32[256]{0}, f32[256]{0}, bf16[128,56,56,256]{3,0,2,1}) fusion(f32[256]{0}, f32[256]{0}, f32[256]{0}, f32[256]{0}, ...(+13)), kind=kOutput, calls=%fused_computation.177, sharding={ {maximal device=0}, {maximal device=0}, {maximal devi...} }
  Allocation type: scoped

	TPU compilation failed
	 [[{{node tpu_compile_succeeded_assert/_2211098028679383727/_435}} = TPUCompileSucceededAssert[_device="/job:worker/replica:0/task:0/device:CPU:0"](TPUReplicate/_compile/_11277858145685444465/_434)]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.

	 [[{{node TPUReplicate/_compile/_11277858145685444465/_434/after_compilation/_436_G6101}} = _Recv[client_terminated=false, recv_device="/job:worker/replica:0/task:0/device:TPU:6", send_device="/job:worker/replica:0/task:0/device:CPU:0", send_device_incarnation=452505474266104386, tensor_name="edge_6943_...ation/_436", tensor_type=DT_FLOAT, _device="/job:worker/replica:0/task:0/device:TPU:6"]()]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.
```

OOM issue??

Oh, this version of XLA complier is something wrong with this special situation, it costs more HBM than usual.

Reducing the batchsize to 512 or even 256 will solve this problem. According to R(Russell), in TFv1.13 the problem will be solved or relief. 