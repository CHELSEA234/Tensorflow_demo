import tensorflow as tf

with tf.variable_scope("foo") as foo_scope:	## foo_scope's name is "foo"
    assert foo_scope.name == "foo"

with tf.variable_scope("bar"):	
    with tf.variable_scope("baz") as other_scope:	## other_scope's name is "bar/baz"
        assert other_scope.name == "bar/baz"
        ## nested naming, it will be hierarchy
        with tf.variable_scope("bay") as foo_scope3:	
            assert foo_scope3.name == "bar/baz/bay"  
        ## this scope is copy of one scope that is named already
        with tf.variable_scope(foo_scope) as foo_scope2:	## foo_scope2's name is "foo"
            assert foo_scope2.name == "foo"  # Not changed.
            