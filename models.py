import tensorflow as tf

def compute_loss(logits, labels):    
    labels = tf.squeeze(tf.cast(labels, tf.int32))     
    cross_entropy =      
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,       labels=labels)    
    cross_entropy_mean = tf.reduce_mean(cross_entropy)    
    tf.add_to_collection('losses', cross_entropy_mean)     
    return tf.add_n(tf.get_collection('losses'),       name='total_loss')    
def compute_accuracy(logits, labels):    
    labels = tf.squeeze(tf.cast(labels, tf.int32))    
    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)    
    predicted_correctly = tf.equal(batch_predictions, labels)    
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly,      tf.float32))    
    return accuracy
def get_learning_rate(global_step, initial_value, decay_steps,             decay_rate):    
    learning_rate = tf.train.exponential_decay(initial_value,      global_step, decay_steps, decay_rate, staircase=True)    
    return learning_rate    
def train(total_loss, learning_rate, global_step, train_vars):     
    optimizer = 
    tf.train.AdamOptimizer(learning_rate)     train_variables = train_vars.split(",")     
    grads = optimizer.compute_gradients(        total_loss,        [v for v in tf.trainable_variables() if v.name in          train_variables]        )    
    train_op = optimizer.apply_gradients(grads,      global_step=global_step)    
    return train_op 
