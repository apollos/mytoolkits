import mypackage_pb2
import copy
import google.protobuf.json_format as j_format
import google.protobuf.text_format as pb_txt_format
from  google.protobuf.descriptor import FieldDescriptor

test_enum = mypackage_pb2.PADDING
print(test_enum.values())

tt = 'SAME'
test_enum2 = test_enum.Value(tt)
print(test_enum2)

print(mypackage_pb2.GenePool().filled)

gen_pool = mypackage_pb2.GenePool()
gen_pool.filled = True
conv1 = gen_pool.conv2d_op_list.add()
conv1.filters = 32

conv1.kernels.extend([3,4])

conv1.strides.extend([5,7, 1])

conv1.paddings = mypackage_pb2.PADDING.Value('SAME')
conv1.trainable = True
conv1.last_op = False


conv2 = gen_pool.conv2d_op_list.add()
conv2.filters = 16

conv2.kernels.extend([1,5])

conv2.strides.extend([5,7, 1])

conv2.paddings = mypackage_pb2.PADDING.Value('VALID')
conv1.trainable = True
conv2.last_op = True


softmax = gen_pool.softmax_cross_entropy_op_list.add()

softmax.weights.extend([1])
softmax.last_op = True
softmax.lable_smoothing = 0

jsfile = j_format.MessageToJson(gen_pool)

with open("a.json", 'w') as f:
    f.write(jsfile)

with open("a.protxt", "w") as f:
    f.write(pb_txt_format.MessageToString(gen_pool, as_utf8=True))

gen_pool2 = mypackage_pb2.GenePool()
with open("b.json", 'r') as f:
    jsfile = f.read()
    j_format.Parse(jsfile, gen_pool2)
print(gen_pool2.conv2d_op_list)

print(list(map( lambda s: (s.name, s.label, s.type), gen_pool2.DESCRIPTOR.fields)))
print(list(map( lambda s: (s.name, s.label, s.type), gen_pool2.conv2d_op_list[1].DESCRIPTOR.fields)))
conv_fields = list(map( lambda s: (s.name, s.label), gen_pool2.conv2d_op_list[1].DESCRIPTOR.fields))
for field in conv_fields:
    if field[1] == FieldDescriptor.LABEL_REPEATED:
        print(getattr(gen_pool2.conv2d_op_list[1], field[0]))
print("===========================")
print(gen_pool2)
print("---------------------------")
gen_pool2.conv2d_op_list[1].ClearField("kernels")
gen_pool2.conv2d_op_list[1].kernels.append(9)
gen_pool2.MergeFrom(gen_pool)
print(gen_pool2)
'''
try:
    gen_pool2.ClearField("conv2d_op_list")
except ValueError as e:
    print("AA" + str(e))
print(gen_pool2)
'''
gen_pool3 = mypackage_pb2.GenePool()
conlst = getattr(gen_pool3, "conv2d_op_list")
print(len(conlst))
'''
t_add_con = conlst.add()
t_add_con.MergeFrom(conv2)
'''
conlst.MergeFrom(gen_pool2.conv2d_op_list)
gen_pool3.filled = True
print(gen_pool3)
print("--------------------------")
gen_pool3.ClearField("filled")
print(gen_pool3)

test_json = '{"filled": true}'
setattr(gen_pool3, 'filled', True)
print(gen_pool3)
print(type(gen_pool3.conv2d_op_list))

a = list(filter(lambda s: s.name == 'filled',  gen_pool2.DESCRIPTOR.fields))
print(a[0].type)

test_json = '{ "filters": 32, "kernels": [ 1, 1 ], "trainable": true}'
cc = gen_pool3.conv2d_op_list.add()
j_format.Parse(test_json, cc)
print(gen_pool3)

print(conv2)
conv2.MergeFrom(cc)
print(conv2)
print("****************************")
#gen_pool2.conv2d_op_list[0].kernels.MergeFrom(gen_pool3.conv2d_op_list[0].kernels)
print(gen_pool2.conv2d_op_list[0].kernels)
gen_pool2.conv2d_op_list[0].kernels.extend(gen_pool3.conv2d_op_list[0].kernels)
print(len(gen_pool2.conv2d_op_list[0].kernels))

print(mypackage_pb2._GENEPOOL.fields_by_name['conv2d_op_list'].message_type.name)