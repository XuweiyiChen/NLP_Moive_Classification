	?A?<?I7@?A?<?I7@!?A?<?I7@	ȒCp????ȒCp????!ȒCp????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?A?<?I7@!\?zz??1?:?? B5@A????(??IM?]~??Y?Oqx???rEagerKernelExecute 0*	??|?5b@2F
Iterator::Model????}ɲ?!?y0,?rI@);ŪA???1??]O?B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatZ?xZ~???!
&f???9@)^??yȔ??1????wc5@:Preprocessing2U
Iterator::Model::ParallelMapV2??C?r???!%?9?+@)??C?r???1%?9?+@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?IbI????!?Gwg?3@)4?Op???1???Q)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice+?򑔄?!l??`W?@)+?򑔄?1l??`W?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorZ???аx?!|?сϸ@)Z???аx?1|?сϸ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipk?SUh ??!]???|?H@)`=?[?w?1???B@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9ȒCp????I`?f? @Q??V?l?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	!\?zz??!\?zz??!!\?zz??      ??!       "	?:?? B5@?:?? B5@!?:?? B5@*      ??!       2	????(??????(??!????(??:	M?]~??M?]~??!M?]~??B      ??!       J	?Oqx????Oqx???!?Oqx???R      ??!       Z	?Oqx????Oqx???!?Oqx???b      ??!       JGPUYȒCp????b q`?f? @y??V?l?V@