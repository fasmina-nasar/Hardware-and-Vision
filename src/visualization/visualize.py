import matplotlib.pyplot as plt

def visualize_data(data, class_name):
  plt.figure(figsize=(10,10))
  for image_batch,label_batch in data.take(1):
    for i in range(12):
     ax=plt.subplot(3,4,i+1)
     plt.imshow(image_batch[i].numpy().astype('uint8'))
     plt.title(class_name[label_batch[i]])
     plt.axis('off')
     plt.show()  
      
def model_performance(var, validation_var, var_name):
  plt.plot(var,label=f'Training {var_name}')
  plt.plot(validation_var, label= f'Validation {var_name}')
  plt.xlabel('epochs')
  plt.ylabel(var_name)
  if var_name == 'loss':
    plt.legend(loc='upper right')
  else:
    plt.legend(loc='lower right')
  plt.title(f'Training and Validation {var_name}')
  plt.show()