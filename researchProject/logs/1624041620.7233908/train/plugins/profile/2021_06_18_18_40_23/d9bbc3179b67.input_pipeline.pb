	I??&@I??&@!I??&@	????S@????S@!????S@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLI??&@???o??1??a?y??A?U??Ά??Iŏ1w-?@Y|?5Z???rEagerKernelExecute 0*	??ʡE?e@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap\??????!?R`6?C@)??+????1?-8?<@:Preprocessing2F
Iterator::ModelB]?P???!?:??A@)?d??1?߇cR(5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat]QJVգ?!X?7?C6@)?t?i???17???V?1@:Preprocessing2U
Iterator::Model::ParallelMapV2P?,?cy??!?+???!*@)P?,?cy??1?+???!*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicej.7값?!?????$@)j.7값?1?????$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip ??G????!?b???qP@)Na?????1G?Z=?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????%~?!? ???@)?????%~?1? ???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?41.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t22.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9????S@ID?0z03P@Q醽?Y=@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???o?????o??!???o??      ??!       "	??a?y????a?y??!??a?y??*      ??!       2	?U??Ά???U??Ά??!?U??Ά??:	ŏ1w-?@ŏ1w-?@!ŏ1w-?@B      ??!       J	|?5Z???|?5Z???!|?5Z???R      ??!       Z	|?5Z???|?5Z???!|?5Z???b      ??!       JGPUY????S@b qD?0z03P@y醽?Y=@