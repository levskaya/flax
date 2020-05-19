# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""NN base modules for JAX."""

import abc
import contextlib
import functools
import hashlib
import inspect
from typing import Any

from . import utils
from . import stochastic
from flax import jax_utils
from flax import serialization
from flax import struct

import jax
from jax import random


_module_stack = utils.CallStack()
_state_stack = utils.CallStack()


class _ModuleFrame:
  """A ModuleFrame the context needed to init or apply a Module.

  In particular, `self.params` is a dictionary where parameters are
  stored (during module init) and read from (during module application).

  When `module.init()` is first called, a new ModuleFrame is created with
  an empty `params` dictionary. When `self.param` is called within that
  module, a new key is added to track that parameter, with the computed
  parameter's initial value.

  When a module calls into a submodule, a new key is added, with a value
  being an empty dictionary. Then that new dictionary is passed in as `params`
  on a new sub-ModuleFrame. That new sub-ModuleFrame keeps track of its parent
  with the `parent` attribute.

  When the whole init process is complete, the top-level ModuleFrame'
  `params` are returned, which contain a nested dictionary of parameters.

  During module application, a similer process happens but this time
  the parameters are only read from.

  Additional attributes on ModuleFrame track context needed to assist error
  handling, shared parameters.

  TODO: Consider elaborating on this last paragraph.
  """

  def __init__(self, name,
               parent=None, params=None, rng=None):
    if params is None:
      params = {}
    self.parent = parent
    self.rng = rng
    self.params = params
    self.name = name
    self._name_counter = 0

  @property
  def is_init(self):
    return self.rng is not None

  @property
  def path(self):
    """Path of the the module scope.

    paths are similar to unix file names (eg. '/module/nested/dense')

    Returns:
      The path of this Module scope.
    """
    if self.parent is None:
      if self.name is None:
        return '/'
      else:
        return '/' + self.name

    path = self.parent.path
    if path[-1] != '/':
      path += '/'
    path += self.name
    return path

  def is_descendent_of(self, frame):
    """Check whether this frame is a descendent of the given frame."""
    if frame is self.parent:
      return True
    elif self.parent:
      return self.parent.is_descendent_of(frame)
    else:
      return False

  def create_name(self):
    name = str(self._name_counter)
    self._name_counter += 1
    return name


def _fold_in_str(rng, data):
  """Fold a string into a jax.random.PRNGKey using its SHA-1 hash."""
  m = hashlib.sha1()
  m.update(data.encode('utf-8'))
  d = m.digest()
  hash_int = int.from_bytes(d[:4], byteorder='big')
  return random.fold_in(rng, hash_int)


class Module:
  """Functional modules."""

  def __new__(cls, *args, **kwargs):
    # record point in module stack that object was instantiated
    # needed for passing shared submodules around as arguments
    instance = object.__new__(cls)
    if _module_stack:
      instance._parent = _module_stack[-1]
    else:
      instance._parent = None # sentinel instead?
    return instance

  def __call__(self, *args, **kwargs):
    if not _module_stack:
      raise ValueError('A Module should only be instantiated directly inside'
                       ' another module.')
    # needed for top-level multi-method module-instance attr inits
    if not self._parent:
      self._parent = _module_stack[-1]

    if not hasattr(self, 'name') or self.name is None:
      self.name = type(self).__name__ + '_' + self._parent.create_name()
    self._check_name(self.name, self._parent)

    if self._parent.is_init and self.name not in self._parent.params:
      rng = _fold_in_str(self._parent.rng, self.name)
      params = {}
      self._parent.params[self.name] = params
    else:  # apply
      if self.name not in self._parent.params:
        raise ValueError(f'No module named {self.name} was created during'
                         ' initialization.')
      params = self._parent.params[self.name]
      rng = None
    frame = _ModuleFrame(self.name, parent=self._parent, rng=rng, params=params)

    with self._with_instance(frame):
      y = self.apply(*args, **kwargs)
    return y

  @abc.abstractmethod
  def apply(self, *args, **kwargs):
    pass

  def init(self, _rng, *args, name=None, **kwargs):
    """Initialize the module parameters.

    Args:
      _rng: the random number generator used to initialize parameters.
      *args: arguments passed to the module's apply function
      name: name of this module.
      **kwargs: keyword arguments passed to the module's apply function
    Returns:
      A pair consisting of the model output and the initialized parameters
    """
    if _module_stack:
      parent = _module_stack[-1]
    else:
      parent = None

    frame = _ModuleFrame(name, rng=_rng, parent=parent)
    with self._with_instance(frame) as instance:
      y = instance.apply(*args, **kwargs)
    return y, frame.params

  def init_by_shape(self, _rng, input_specs, *args, name=None, **kwargs):
    """Initialize the module parameters.

    This method will initialize the module parameters without computation.
    Initializer functions can depend on the shape but not the value of inputs.

    Args:
      _rng: the random number generator used to initialize parameters.
      input_specs: an iterable of (shape, dtype) pairs specifying the inputs
      *args: arguments passed to the module's apply function
      name: name of this module.
      **kwargs: keyword arguments passed to the module's apply function
    Returns:
      A pair consisting of the model output and the initialized parameters
    Example:
      ```
      input_shape = (batch_size, image_size, image_size, 3)
      model_output, initial_params = model.init_by_shape(jax.random.PRNGKey(0),
                                      input_specs=[(input_shape, jnp.float32)])
      ```
    """
    stochastic_rng = None
    try:
      stochastic_rng = stochastic.make_rng()
    except ValueError:
      # Either there is no stochastic scope or the current
      # scope is invalid due to another jax transformation.
      # In both cases we should not try to lift the stochastic
      # scope into the lazy evaluation
      pass

    def lazy_init(*inputs):
      def init_fn():
        return self.init(_rng, *(inputs + args), name=name, **kwargs)
      if stochastic_rng is not None:
        # Create a new stochastic scope inside the lazy evalution
        # this way we can use a stochastic scope in combination
        # with init_by_shape.
        with stochastic.stochastic(stochastic_rng):
          return init_fn()
      else:
        return init_fn()
    return jax_utils.partial_eval_by_shape(lazy_init, input_specs)

  def call(self, params, *args, name=None, **kwargs):
    """Evaluate the module with the given parameters.

    Args:
      params: the parameters of the module. Typically, inital parameter values
        are constructed using `Module.init` or `Module.init_by_shape`.
      *args: arguments passed to the module's apply function
      name: name of this module.
      **kwargs: keyword arguments passed to the module's apply function
    Returns:
      The output of the module's apply function.
    """
    if _module_stack:
      parent = _module_stack[-1]
    else:
      parent = None

    frame = _ModuleFrame(name, params=params, parent=parent)
    with self._with_instance(frame) as instance:
     y = instance.apply(*args, **kwargs)
    return y

  def param(self, name, shape, initializer):
    """Defines a parameter within the module's apply function.

    Args:
      name: The name of the parameter.
      shape: The shape of the parameter. If None the param be any type.
      initializer: An initializer function
                   taking an RNG and the shape as arguments.
    Returns:
      The value of the parameter.
    """
    frame = self._frame
    if frame.is_init:
      if name in frame.params:
        raise ValueError(
            "Name '%s' was already used for another parameter." % name)
      key = _fold_in_str(frame.rng, name)
      frame.params[name] = initializer(key, shape)
    if name not in frame.params:
      raise ValueError("Parameter with name '%s' does not exist." % name)
    param = frame.params[name]
    if shape is not None and param.shape != shape:
      raise ValueError(
          'Existing shape {} differs from requested shape {}'.format(
              param.shape, shape))
    return param

  def get_param(self, name):
    """Retrieves a parameter within the module's apply function.

    Args:
      name: The name of the parameter.
    Returns:
      The value of the parameter.
    """
    frame = self._frame
    if name not in frame.params:
      raise ValueError("Parameter with name '%s' does not exist." % name)
    return frame.params[name]

  def state(self, name, shape=None, initializer=None, collection=None):
    """Declare a state variable within the module's apply function.

    A state variable has an attribute value which can be updated by simply
    assigning a value to it. For example::

      class Example(nn.Module):
        def apply(self, inputs, decay=0.9):
          ema = self.state('ema', inputs.shape, initializers.zeros)
          ema.value = decay * ema.value + (1 - decay) * inputs
          return inputs

    By default Modules are stateless. See `flax.nn.stateful` to enable stateful
    computations.

    Args:
      name: the name of the state variable.
      shape: optional shape passed to the initializer (default: None)
      initializer: optional initializer function
        taking an RNG and the shape as arguments.
      collection: optional `flax.nn.Collection` used to store the state.
        By default the state collection passed to the `nn.stateful` context is
        used.
    Returns:
      An instance of ModuleState.
    """
    _top_frame('state')
    if collection is None:
      collection = get_state()
    state = ModuleState(collection, name)
    # find the frames that are in init mode
    init_frames = [f for f in _module_stack if f.is_init]
    if initializer is not None and init_frames:
      # use the closest frame that is initializing to get an rng
      init_frame = init_frames[-1]
      init_frame.rng, key = random.split(init_frame.rng)
      init_value = initializer(key, shape)
      state.value = init_value
    return state

  def is_stateful(self):
    return is_stateful()

  def is_initializing(self):
    _top_frame('is_initializing')
    return self._frame.is_init

  @contextlib.contextmanager
  def _with_instance(self, frame):  # TODO: rename this
    """Private constructor for Module.

    A module instance is constructed using a scope and is tied to a _ModuleFrame
    This way the methods on the Module instance can rely on the _ModuleFrame
    being available.

    Args:
      frame: an instance of _ModuleFrame
    Yields:
      An instance of Module
    """
    self._frame = frame
    with _module_stack.frame(self._frame):
      yield self

  @classmethod
  def _check_name(cls, name, parent):
    """Check whether the name of the module is valid within the parent scope."""
    if name is not None:
      if not isinstance(name, str):
        raise ValueError('Name must be a string.')
      if '/' in name or ':' in name:
        raise ValueError('Name should not contain slashes or colons.')
      if name in parent.params:
        raise ValueError(f'A module with named "{name}" already exists.')


def module(fun):
  """Convert a function into the apply method of a new Module.

  This is convenient shortcut for writing higher level modules that don't need
  access to `self` for creating parameters directly.

  Example usage::

    @nn.module
    def DenseLayer(x, features):
      x = flax.nn.Dense(x, features)
      x = flax.nn.relu(x)
      return x

  This is exactly equivalent to defining the following `nn.Module` subclass::

    class DenseLayer(nn.Module):
      def apply(self, x, features):
        x = flax.nn.Dense(x, features)
        x = flax.nn.relu(x)
        return x

  Args:
    fun: the function to convert.
  Returns:
    New Module subclass.
  """
  @functools.wraps(fun)
  def apply(self, *args, **kwargs):
    del self  # unused
    return fun(*args, **kwargs)
  return type(fun.__name__, (Module,), dict(apply=apply))


class ModuleState():
  """Tracks a state variable.

  ModuleState instances should not be created directly. See `Module.state` on
  how to create state variables inside modules.
  """

  def __init__(self, collection, name):
    self._collection = collection
    self._name = name

  def _get_state_dict(self):
    state_dict = self._collection.retrieve(default={})
    assert isinstance(state_dict, dict)
    return state_dict

  @property
  def name(self):
    return self._name

  @property
  def value(self):
    state_dict = self._get_state_dict()
    if self._name not in state_dict:
      raise ValueError(f'No state variable named `{self._name}` exists.')
    return state_dict[self._name]

  @value.setter
  def value(self, v):
    state_dict = self._get_state_dict()
    state_dict[self._name] = v
    self._collection.store(state_dict)


@contextlib.contextmanager
def stateful(state=None, mutable=True):
  """A context manager for stateful computations.

  Module's that use the `Module.state` by default store state inside the
  `Collection` specified by the (innermost) `nn.stateful` context manager.

  Typically stateful is used in 3 different modes:

  1. During init no existing state is available and the stateful context creates
     a new state collection.
  2. During training the state is passed to `nn.stateful` and the new state
     is returned which will contain the updated state.
  3. During evaluation the state is passed with `mutable=False` such that the
     model can retrieve the state but is not allowed to mutate it.

  Example::

    class MyModel(nn.Module):
      def apply(self, x):
        x = nn.Dense(x, 12)
        x = nn.BatchNorm(x)
        return x

    with nn.stateful() as state:
      _, initial_params = MyModel.init(rng, x)
      model = nn.Model(MyModel, initial_params)

    with nn.stateful(state) as new_state:
      model(x2)

    with nn.stateful(new_state, mutable=False):
      evaluate_model(model)

  Args:
    state: a `flax.nn.Collection` containing the current state.
      By default a new collection will be created.
    mutable: If true the state will be mutable otherwise it will be frozen.
  Yields:
    A `flax.nn.Collection` containing the new state.
  """
  if state is None:
    state = Collection()
  if mutable:
    with state.mutate() as new_state:
      with _state_stack.frame(new_state):
        yield new_state
  else:
    with _state_stack.frame(state):
      yield state


def is_stateful():
  """Returns true if a stateful scope is currently active (see `flax.nn.stateful`)."""
  return bool(_state_stack)


def get_state():
  if not _state_stack:
    raise ValueError('Use the flax.nn.stateful context manager to enable'
                     ' stateful computations.')
  return _state_stack[-1]


def _top_frame(call_name):
  if not _module_stack:
    raise ValueError('%s should only be used inside a '
                     'module\'s apply function.' % call_name)
  return _module_stack[-1]


@struct.dataclass
class Model:
  """A Model contains the model paramaters, state and definition."""

  module: Module = struct.field(pytree_node=False)
  params: Any

  def __call__(self, *args, **kwargs):
    return self.module.call(self.params, *args, **kwargs)

  def __getattr__(self, name):
    value = getattr(self.module, name)
    if inspect.isclass(value) and issubclass(value, Module):
      def wrapper(*args, **kwargs):
        return value.call(self.params, *args, **kwargs)
      return wrapper
    raise AttributeError(f'No attribute named "{name}".')

  def __hash__(self):
    # Jax will call hash when model is passed to a function transform.
    # the compiled function should not be shared among model instances because
    # it closes over the specific parameters of this model instance.
    return id(self)


class Collection:
  """A collection of tensors useful for tracking state.

  A Collection can be used to associate data with the application of a Module.
  For example a collection can be used to collect activations across modules.
  Another common use case for collections is to track internal state.
  For example, the running averages in BatchNorm can be stored in a collection.

  Attributes:
    state: the initial state by default an empty collection is created.
  """

  def __init__(self, state=None):
    if state is None:
      state = {}
    self.state = state
    # The anchor is used to determine the prefix of the collection.
    # This way we can create/nest collections inside modules.
    self._anchor = _module_stack[-1] if _module_stack else None

    self._mutable = False
    self._master_level = None
    self._root = None

  def as_dict(self):
    """Returns a dictionary with module paths as keys and the stored values.

    Returns:
      The stored values as a dictionary.
    """
    return self.state.copy()

  def __getitem__(self, key):
    return self.state[key]

  @contextlib.contextmanager
  def mutate(self):
    # pylint: disable=protected-access
    new_col = jax.tree_map(lambda x: x, self)  # clone the collection
    new_col._mutable = True
    new_col._master_level = utils._trace_level(utils._current_trace())
    try:
      yield new_col
    finally:
      new_col._mutable = False

  def retrieve(self, default=None):
    """Retrieves a value from the Collection.

    This functions should only be called with the apply function of a module.
    Args:
      default: The default returned when nothing is stored (default: None)
    Returns:
      The value previously stored in the collection.
    """
    _top_frame('retrieve')
    path = self._current_path()
    return self.state.get(path, default)

  def store(self, value):
    """Stores a value in the Collection.

    This functions should only be called with the apply function of a module.
    Args:
      value: The value to be stored in the collection
    Returns:
      The previous value stored in the collection or None.
    """
    frame = _top_frame('store')
    if not self._mutable:
      raise ValueError('Collection is not mutable. Use the `mutate` method to'
                       ' create a mutable copy.')
    # Use the Jax TraceMaster to determine if a Collection is modified from
    # inside a nested jax transformation.
    # In this case, we throw an error because transforming a stateful function
    # is ill-defined (eg. what does vmap of BatchNorm do?).
    # TODO(jheek): Add doc guide on combining jax transforms and state.
    # TODO(jheek): Should some transformations be excempt from this error?
    value_level = utils._level_of_value(value)
    if value_level > self._master_level:
      raise ValueError('Stateful operations are not allowed when the Collection'
                       ' is created outside of the current Jax transformation')

    # The root of a Collection is the first module scope that gets created
    # inside the mutate scope of the Collection. By allowing only one unique
    # root scope, we guarantee that state is not accidentally shared
    # between different models. When a user specifies an explicit name we can
    # distinguish models and a collection can have multiple roots.
    if frame == self._anchor:
      # Example:
      # with nn.Collection.mutate() as coll:
      #   coll.store(1)
      raise ValueError('State should be stored from within a module.'
                       ' Consider using the value directly instead of'
                       ' storing it in a Collection.')
    if not frame.is_descendent_of(self._anchor):
      # edge case where the Collection cannot capture the scope of a shared Module
      # See test_collection_store_fails_if_out_of_scope in nn_test.py
      raise ValueError('Trying to capture state outside the scope of this Collection.'
                       ' Most likely due to passing around a shared Module.')
    root = self._find_root(frame)
    if self._root is None:
      self._root = root
    elif self._root != root:
      if self._root.name is None or root.name is None:
        # In the following examples, should the two calls to `StatefulModule` share state or not?
        # Because it's ambiguous, we throw an error and require the user to explicitly separate state
        # by giving each instance a separate name, or to explicitly pass the same name
        # in order to share state.
        # with nn.statefull(state) as new_state:
        #   StatefulModule.call(params)
        #   StatefulModule.call(params2)
        raise ValueError('When multiple top-level module calls use a Collection'
                         ' each top-level module should have a name.')
    path = self._current_path()
    old_value = self.state.get(path, None)
    self.state[path] = value
    return old_value

  def _find_root(self, frame):
    """Find the root frame with respect to the anchor.

    The root frame is defined as the child of anchor
    that is an ancestor of frame.
    The root is used to verify that a Collection does not
    have multiple unnamed roots.

    Args:
      - frame: the frame of which we want to know the root
    Returns:
      The root of the given frame.
    """
    assert frame.is_descendent_of(self._anchor)
    root = frame
    while root.parent != self._anchor:
      root = root.parent
    return root

  def _current_path(self):
    """"The relative path from the currently active module scope to the root of the collection.

    For example: If a collection is created in the path '/module/nested' and
    something is stored by a module with the path '/module/nested/block/conv'
    the key in the collection dict will be '/block/conv'.

    Returns:
      the relative path of the active module scope.
    """
    frame = _module_stack[-1]
    assert frame.is_descendent_of(self._anchor)
    path = _module_stack[-1].path
    if self._anchor is not None and self._anchor.path != '/':
      prefix = self._anchor.path
      assert prefix == path[:len(prefix)]
      return path[len(prefix):]
    else:
      return path

def iterate_collection(collection):
  # jax iterates through pytrees for each argument/return value of a functional
  # transformations. When the collection is mutable we throw an error this way
  # we avoid silent errors due to impurity of a traced function.
  if collection._mutable:  # pylint: disable=protected-access
    raise ValueError('A mutable collection should not be transformed by Jax.')
  meta = (type(collection), collection._anchor)  # pylint: disable=protected-access
  return (collection.state,), meta


def collection_from_iterable(meta, state):
  ty, anchor = meta
  coll = ty(state[0])
  coll._anchor = anchor  # pylint: disable=protected-access
  return coll

# make sure a collection is traced.
jax.tree_util.register_pytree_node(Collection,
                                   iterate_collection,
                                   collection_from_iterable)


def _collection_state_dict(collection):
  return collection.as_dict()


def _collection_from_state_dict(_, state):
  return Collection(state)


serialization.register_serialization_state(
    Collection, _collection_state_dict, _collection_from_state_dict)
