	?켍͎.@?켍͎.@!?켍͎.@	??<?q6@??<?q6@!??<?q6@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?켍͎.@kb???? @1,?PO?@A?{?O???IV?@?)6@Y?? k?n@rEagerKernelExecute 0*	?x?&1??@2F
Iterator::Model'?|??	@!??hX@)Sͬ???	@1@ ?3o)X@:Preprocessing2U
Iterator::Model::ParallelMapV2?4?;???!3?['?H??)?4?;???13?['?H??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??*Q????!e7??@y??)i? ?w???1?H????:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice???0X??!Dk>??)???0X??1Dk>??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??/??L??!?<???3??)M??ӀA??1????(??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipDOʤ?6??!????@)?б?J|?1S8ܝ????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorR~R???x?!٘?|rr??)R~R???x?1٘?|rr??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 22.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.high"?44.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2t13.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??<?q6@I
?=Z?)M@Q[???:3@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	kb???? @kb???? @!kb???? @      ??!       "	,?PO?@,?PO?@!,?PO?@*      ??!       2	?{?O????{?O???!?{?O???:	V?@?)6@V?@?)6@!V?@?)6@B      ??!       J	?? k?n@?? k?n@!?? k?n@R      ??!       Z	?? k?n@?? k?n@!?? k?n@b      ??!       JGPUY??<?q6@b q
?=Z?)M@y[???:3@