	???0?? @???0?? @!???0?? @	af%??v.@af%??v.@!af%??v.@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL???0?? @?>+??1tѐ?(?@AE?4f??Ivnڌ??@YJ??1??rEagerKernelExecute 0*	??Q?Z?@2U
Iterator::Model::ParallelMapV2Ɔn?J??!???j?R@)Ɔn?J??1???j?R@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice}?R??c??!?gOu??-@)}?R??c??1?gOu??-@:Preprocessing2F
Iterator::Model?3??????!??oW|?S@)%Ί??>??1<?ڔ?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?U1?~??!??kԁ?@)?[?~l???1^c?9 @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap,H3Mg??!t^?ӻ?0@){????1U?Ǵ
 @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?9}=_??!??@??4@)?V?????1? ??R???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??????u?!?t??<???)??????u?1?t??<???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 15.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?29.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t18.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9af%??v.@I?q?H@Qm4?3??A@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?>+???>+??!?>+??      ??!       "	tѐ?(?@tѐ?(?@!tѐ?(?@*      ??!       2	E?4f??E?4f??!E?4f??:	vnڌ??@vnڌ??@!vnڌ??@B      ??!       J	J??1??J??1??!J??1??R      ??!       Z	J??1??J??1??!J??1??b      ??!       JGPUYaf%??v.@b q?q?H@ym4?3??A@