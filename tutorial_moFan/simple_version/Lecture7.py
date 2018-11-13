import tensorflow as tf

state = tf.Variable(10, name='counter')			# the type is variables.Variable
print (state.name)			#the result will be counter:0
one = tf.constant(1)			# the type is Tensor
print ('the constant result is',(one),' you can\'t print out tensor result direclty without session')
with tf.Session() as sess:
    print ((sess.run(one)), 'using session is right way')
    A = sess.run(one)
print ("you can get result from session: ", A)


# all tf operator's result only could be seen in session
new_value = tf.add(state, one)
print (new_value)			# won't output anything
update = tf.assign(state, tf.add(state, one))
print ('the update" result',(update))		# won't output anything, you still use run

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print ("all tf operator's result only could be seen in session")
print (sess.run(new_value))		
print (sess.run(update))
sess.close()



init = tf.global_variables_initializer()  

with tf.Session() as sess:
	sess.run(init)
	print ('if I just run new value and add function ', sess.run(tf.add(state,one)))

with tf.Session() as sess:
    sess.run(init)	## print sess.run(init), if you output, it would be none
    print ("--------------")
    for _ in range(3):
        sess.run(update)		#call updata, the newvalue will be assigned 
       							#you should run handler first, the value will be updated
        print(sess.run(state))		#put the sess pointer on the state, it will be shown, print(state) won't work
