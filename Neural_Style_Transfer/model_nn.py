import tensorflow as tf
from save_image import save_image

def model_nn(sess, input_image, model, train_step, J, J_content, J_style, output_image_path="output/", num_iterations = 200):
    
    # Initialize global variables (you need to run the session on the initializer)
    sess.run(tf.compat.v1.global_variables_initializer())
    
    # Run the noisy input image (initial generated image) through the model. Use assign().
    model["input"].assign(input_image)
    
    for i in range(num_iterations):
    
        # Run the session on the train_step to minimize the total cost
        sess.run(train_step)
        
        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])

        # Print every 20 iteration.
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            # save current generated image in the "/output" directory
            save_image( output_image_path+ str(i) + ".png", generated_image)
    
    # save last generated image
    save_image(output_image_path + "generated_image.jpg", generated_image)
    
    return generated_image