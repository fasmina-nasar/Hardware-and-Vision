def predict_image(model, test_data, class_names):
  # Get the first image and label from the test set
  for images_batch,labels_batch in test_data.take(1):
    first_image=images_batch[0].numpy().astype('uint8')
    first_label=labels_batch[0].numpy()

    # Display the first image and its actual label
    print('first image to predict')
    plt.imshow(first_image)
    print('actual_label :',class_names[first_label])

    # Make a prediction on the first image
    batch_prediction=model.predict(images_batch)
    print("predicted label:",class_names[np.argmax(batch_prediction[0])])

    # Display the predicted label
    plt.axis('off')
    plt.show()
