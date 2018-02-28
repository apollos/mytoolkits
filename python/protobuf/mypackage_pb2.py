# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mypackage.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='mypackage.proto',
  package='dlOp',
  syntax='proto3',
  serialized_pb=_b('\n\x0fmypackage.proto\x12\x04\x64lOp\"\x91\x01\n\tConv2D_op\x12\x0f\n\x07\x66ilters\x18\x01 \x01(\x05\x12\x0f\n\x07kernels\x18\x02 \x03(\x05\x12\x0f\n\x07strides\x18\x03 \x03(\x05\x12\x1f\n\x08paddings\x18\x04 \x01(\x0e\x32\r.dlOp.PADDING\x12\x0c\n\x04\x62ias\x18\x05 \x01(\x08\x12\x11\n\ttrainable\x18\x06 \x01(\x08\x12\x0f\n\x07last_op\x18\x07 \x01(\x08\"s\n\x15Softmax_Cross_Entropy\x12\x1f\n\x04type\x18\x01 \x01(\x0e\x32\x11.dlOp.LOSS_LOGITS\x12\x0f\n\x07weights\x18\x02 \x03(\x02\x12\x0f\n\x07last_op\x18\x03 \x01(\x08\x12\x17\n\x0flable_smoothing\x18\x04 \x01(\x05\"\x87\x01\n\x08GenePool\x12\x0e\n\x06\x66illed\x18\x01 \x01(\x08\x12\'\n\x0e\x63onv2d_op_list\x18\x02 \x03(\x0b\x32\x0f.dlOp.Conv2D_op\x12\x42\n\x1dsoftmax_cross_entropy_op_list\x18\x10 \x03(\x0b\x32\x1b.dlOp.Softmax_Cross_Entropy*\x1e\n\x07PADDING\x12\x08\n\x04SAME\x10\x00\x12\t\n\x05VALID\x10\x01*\x1f\n\x07POOLING\x12\x0b\n\x07\x41VERAGE\x10\x00\x12\x07\n\x03MAX\x10\x01*I\n\nACTIVATION\x12\x08\n\x04RELU\x10\x00\x12\x08\n\x04SELU\x10\x01\x12\x0c\n\x08SOFTPLUS\x10\x02\x12\x0c\n\x08SOFTSIGN\x10\x03\x12\x0b\n\x07SIGMOID\x10\x04*!\n\x13WEIGHT_INITIALIZERS\x12\n\n\x06xavier\x10\x00*\x1f\n\x13\x42IASES_INITIALIZERS\x12\x08\n\x04zero\x10\x00*\"\n\x0fLOSS_PREDICTION\x12\x0f\n\x0bpredictions\x10\x00*\x19\n\x0bLOSS_LOGITS\x12\n\n\x06logits\x10\x00\x62\x06proto3')
)

_PADDING = _descriptor.EnumDescriptor(
  name='PADDING',
  full_name='dlOp.PADDING',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='SAME', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VALID', index=1, number=1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=428,
  serialized_end=458,
)
_sym_db.RegisterEnumDescriptor(_PADDING)

PADDING = enum_type_wrapper.EnumTypeWrapper(_PADDING)
_POOLING = _descriptor.EnumDescriptor(
  name='POOLING',
  full_name='dlOp.POOLING',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='AVERAGE', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MAX', index=1, number=1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=460,
  serialized_end=491,
)
_sym_db.RegisterEnumDescriptor(_POOLING)

POOLING = enum_type_wrapper.EnumTypeWrapper(_POOLING)
_ACTIVATION = _descriptor.EnumDescriptor(
  name='ACTIVATION',
  full_name='dlOp.ACTIVATION',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='RELU', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SELU', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SOFTPLUS', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SOFTSIGN', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SIGMOID', index=4, number=4,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=493,
  serialized_end=566,
)
_sym_db.RegisterEnumDescriptor(_ACTIVATION)

ACTIVATION = enum_type_wrapper.EnumTypeWrapper(_ACTIVATION)
_WEIGHT_INITIALIZERS = _descriptor.EnumDescriptor(
  name='WEIGHT_INITIALIZERS',
  full_name='dlOp.WEIGHT_INITIALIZERS',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='xavier', index=0, number=0,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=568,
  serialized_end=601,
)
_sym_db.RegisterEnumDescriptor(_WEIGHT_INITIALIZERS)

WEIGHT_INITIALIZERS = enum_type_wrapper.EnumTypeWrapper(_WEIGHT_INITIALIZERS)
_BIASES_INITIALIZERS = _descriptor.EnumDescriptor(
  name='BIASES_INITIALIZERS',
  full_name='dlOp.BIASES_INITIALIZERS',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='zero', index=0, number=0,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=603,
  serialized_end=634,
)
_sym_db.RegisterEnumDescriptor(_BIASES_INITIALIZERS)

BIASES_INITIALIZERS = enum_type_wrapper.EnumTypeWrapper(_BIASES_INITIALIZERS)
_LOSS_PREDICTION = _descriptor.EnumDescriptor(
  name='LOSS_PREDICTION',
  full_name='dlOp.LOSS_PREDICTION',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='predictions', index=0, number=0,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=636,
  serialized_end=670,
)
_sym_db.RegisterEnumDescriptor(_LOSS_PREDICTION)

LOSS_PREDICTION = enum_type_wrapper.EnumTypeWrapper(_LOSS_PREDICTION)
_LOSS_LOGITS = _descriptor.EnumDescriptor(
  name='LOSS_LOGITS',
  full_name='dlOp.LOSS_LOGITS',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='logits', index=0, number=0,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=672,
  serialized_end=697,
)
_sym_db.RegisterEnumDescriptor(_LOSS_LOGITS)

LOSS_LOGITS = enum_type_wrapper.EnumTypeWrapper(_LOSS_LOGITS)
SAME = 0
VALID = 1
AVERAGE = 0
MAX = 1
RELU = 0
SELU = 1
SOFTPLUS = 2
SOFTSIGN = 3
SIGMOID = 4
xavier = 0
zero = 0
predictions = 0
logits = 0



_CONV2D_OP = _descriptor.Descriptor(
  name='Conv2D_op',
  full_name='dlOp.Conv2D_op',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='filters', full_name='dlOp.Conv2D_op.filters', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='kernels', full_name='dlOp.Conv2D_op.kernels', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='strides', full_name='dlOp.Conv2D_op.strides', index=2,
      number=3, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='paddings', full_name='dlOp.Conv2D_op.paddings', index=3,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='bias', full_name='dlOp.Conv2D_op.bias', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='trainable', full_name='dlOp.Conv2D_op.trainable', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='last_op', full_name='dlOp.Conv2D_op.last_op', index=6,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=26,
  serialized_end=171,
)


_SOFTMAX_CROSS_ENTROPY = _descriptor.Descriptor(
  name='Softmax_Cross_Entropy',
  full_name='dlOp.Softmax_Cross_Entropy',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='dlOp.Softmax_Cross_Entropy.type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='weights', full_name='dlOp.Softmax_Cross_Entropy.weights', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='last_op', full_name='dlOp.Softmax_Cross_Entropy.last_op', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='lable_smoothing', full_name='dlOp.Softmax_Cross_Entropy.lable_smoothing', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=173,
  serialized_end=288,
)


_GENEPOOL = _descriptor.Descriptor(
  name='GenePool',
  full_name='dlOp.GenePool',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='filled', full_name='dlOp.GenePool.filled', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='conv2d_op_list', full_name='dlOp.GenePool.conv2d_op_list', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='softmax_cross_entropy_op_list', full_name='dlOp.GenePool.softmax_cross_entropy_op_list', index=2,
      number=16, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=291,
  serialized_end=426,
)

_CONV2D_OP.fields_by_name['paddings'].enum_type = _PADDING
_SOFTMAX_CROSS_ENTROPY.fields_by_name['type'].enum_type = _LOSS_LOGITS
_GENEPOOL.fields_by_name['conv2d_op_list'].message_type = _CONV2D_OP
_GENEPOOL.fields_by_name['softmax_cross_entropy_op_list'].message_type = _SOFTMAX_CROSS_ENTROPY
DESCRIPTOR.message_types_by_name['Conv2D_op'] = _CONV2D_OP
DESCRIPTOR.message_types_by_name['Softmax_Cross_Entropy'] = _SOFTMAX_CROSS_ENTROPY
DESCRIPTOR.message_types_by_name['GenePool'] = _GENEPOOL
DESCRIPTOR.enum_types_by_name['PADDING'] = _PADDING
DESCRIPTOR.enum_types_by_name['POOLING'] = _POOLING
DESCRIPTOR.enum_types_by_name['ACTIVATION'] = _ACTIVATION
DESCRIPTOR.enum_types_by_name['WEIGHT_INITIALIZERS'] = _WEIGHT_INITIALIZERS
DESCRIPTOR.enum_types_by_name['BIASES_INITIALIZERS'] = _BIASES_INITIALIZERS
DESCRIPTOR.enum_types_by_name['LOSS_PREDICTION'] = _LOSS_PREDICTION
DESCRIPTOR.enum_types_by_name['LOSS_LOGITS'] = _LOSS_LOGITS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Conv2D_op = _reflection.GeneratedProtocolMessageType('Conv2D_op', (_message.Message,), dict(
  DESCRIPTOR = _CONV2D_OP,
  __module__ = 'mypackage_pb2'
  # @@protoc_insertion_point(class_scope:dlOp.Conv2D_op)
  ))
_sym_db.RegisterMessage(Conv2D_op)

Softmax_Cross_Entropy = _reflection.GeneratedProtocolMessageType('Softmax_Cross_Entropy', (_message.Message,), dict(
  DESCRIPTOR = _SOFTMAX_CROSS_ENTROPY,
  __module__ = 'mypackage_pb2'
  # @@protoc_insertion_point(class_scope:dlOp.Softmax_Cross_Entropy)
  ))
_sym_db.RegisterMessage(Softmax_Cross_Entropy)

GenePool = _reflection.GeneratedProtocolMessageType('GenePool', (_message.Message,), dict(
  DESCRIPTOR = _GENEPOOL,
  __module__ = 'mypackage_pb2'
  # @@protoc_insertion_point(class_scope:dlOp.GenePool)
  ))
_sym_db.RegisterMessage(GenePool)


# @@protoc_insertion_point(module_scope)