def inference(images, is_training=False):     
    #
     # All the code before fc7 are not modified.     
    #
     fc7 = _fully_connected(fc6, 4096, name="fc7")     
     if is_training:         
         fc7 = tf.nn.dropout(fc7, keep_prob=0.5)     
         fc8 = _fully_connected(fc7, 37, name='fc8-pets', relu=False)     
         return fc8