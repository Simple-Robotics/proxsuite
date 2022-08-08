//
// Copyright (c) 2022 INRIA
//
#ifndef proxsuite_python_helpers_hpp
#define proxsuite_python_helpers_hpp

#define PROXSUITE_PYTHON_EIGEN_READWRITE(class, field_name, doc)               \
  def_property(                                                                \
    #field_name,                                                               \
    [](class& self) { return self.field_name; },                               \
    [](class& self, const decltype(class ::field_name)& value) {               \
      self.field_name = value;                                                 \
    },                                                                         \
    doc)

#endif // ifndef proxsuite_python_helpers_hpp
