from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import  ReduceLROnPlateau
import numpy as np
import hyperparms
import dataset
import model
import argparse
import seaborn as sns
from datetime import datetime
import io




# input_shape = (512)
# config = hyperparams.Config()
# input = Input(input_shape)
# op, att_scores = model._Model(config)(input)
# output = Dense(1, activation='linear')(op)
#
# _model = Model(inputs = input, outputs=output)
#
# _model.summary()
import io


class Train():

    '''
        This class gives us useful methods to train our model

        Attributes:

          optimizer: the optimizer that we want to use. Any one from tf.keras.optimizers
          loss_function: Loss function to be used for training. Any one from tf.keras.losses
          metric: List of metrics we want to test our model on . Any one from tf.keras.losses
          train_data: Training data used for training our model
          test_data: Test Data used for training our model
          batch_size: The batch size for our dataset
          learning_rate: The learning rate to be used for training
          epochs: Number of epochs to train our model for

    '''

    def __init__(self, args, train_data, test_data, batch_size):

        '''Initializes variables'''

        self.args = args
        self.optimizer = tf.keras.optimizers.get(args.optimizer)
        self.loss_function = args.loss_function
        self.metric = args.metric
        self.train_dataset = train_data
        self.test_dataset = test_data
        self.batch_size = batch_size
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        self.problem_id = args.problem_id
        self.config = hyperparms.Config(self.args.num_heads, self.args.num_layers, self.args.emb_dim, self.args.seq_length, self.args.vocab_size, self.args.head_size, self.args.pos_embedding, self.args.agg_method, self.args.pos_embedding_type)

    def create_model(self):

      '''Creates a Model object and returns it'''

      config = self.config
      if(config['agg_method'] == 'TOKEN'):
          input_shape = (config['seq_length'] + 1)
      else:
          input_shape = (config['seq_length'])

      input = Input(input_shape, dtype='int64')
      op, att_scores = model._Model(config['emb_dim'], config['seq_length'], config['vocab_size'], config['pos_embedding'], config['num_heads'], config['head_size'], config['agg_method'], config['pos_embedding_type'], config['num_att_layers'])(input)
      output = Dense(1, activation='linear')(op)

      att_model = Model(inputs = input, outputs=output)
      return att_model

    def get_lr_range(self, total_epoch = 3, show_plot = True):

        '''
          This method helps us to select a appropriate lr for our training

          Using a custom call back , we train our model for 3 epochs and change the
          learning rate after each batch update. At the end the plot will tell us the
          learning rate range between which our loss was reducing.

        '''
        model = self.create_model()
        lr = []
        losses = []



        class LrCallback(tf.keras.callbacks.Callback):

            '''
              Custom call back for getting our learning rate range

              Attributes:
                train_data: Training data used for training our model
                batch_size: The batch size for our dataset
                total_epoch: total epochs to runb our model

            '''

            def __init__(self,train_data,batch_size, total_epoch = 3):

                '''Initilaizes variables'''

                self.X_train = train_data
                self.batch_size = batch_size
                self.total_epoch = total_epoch
                self._chief_worker_only = None


            def on_train_begin(self, logs={}):
                ## on begin of training, we are creating a instance varible called history
                ## it is a dict with keys [loss, acc, val_loss, val_acc]

                self.history={'loss': [],'acc': [],'val_loss': [],'val_acc': [],'AUC':[],'val_AUC':[]}

            def on_epoch_end(self, epoch, logs={}):

                pass

            def on_batch_begin(self,batch,logs):

                pass


            def on_batch_end(self,batch,logs):

                '''
                  Here we will try to find out our learning rate range

                  We will try to increase our learning rate slowly after each batch update,
                  and will store the loss after each batch along with the learning rate.
                  Plotting these two values will give us an idea about our learning rate
                '''




                '''getting loss after batch has ended'''
                l = logs['loss']
                '''getting the learning rate corresponding to the loss'''
                lr.append(K.get_value(self.model.optimizer.lr))
                losses.append(l)

                '''we will start from small lr e-10 to e10 and check our plot for these values'''
                start_lr = self.config['lr_range'][0]
                end_lr = self.config['lr_range'][1]

                '''Number of batch updates in each epoch'''
                #step_size = (len(self.X_train) // self.batch_size)
                step_size = len(self.X_train)
                '''total batch updates across all epochs'''
                iter = (self.total_epoch*step_size)
                '''used to increase our lr exponentialy'''
                LRmult = (end_lr/start_lr)**(1/iter)
                '''calculating new lr'''
                new_lr = K.get_value(self.model.optimizer.lr)*LRmult
                '''setting new lr for our next batch'''
                K.set_value(self.model.optimizer.lr, new_lr)
                #print('lr:',K.get_value(self.model.optimizer.lr))



        lr_callback = LrCallback(self.train_dataset, self.batch_size)

        self.optimizer.learning_rate = self.config['lr_range'][0]
        model = self.create_model()
        model.compile(optimizer= self.optimizer, loss=self.loss_function, metrics = self.metric)
        model.fit(self.train_dataset, epochs = self.total_epoch, validation_data = self.test_dataset, steps_per_epoch = len(self.train_dataset), callbacks=[lr_callback])

        if(show_plot):
            plt.plot(lr, losses)
            plt.xscale("log")
            plt.xlabel("Learning Rate (Log Scale)")
            plt.ylabel("Loss")
            plt.show()

        return lr, losses


    def train_model(self):

      '''
        This function trains our model using the parmeters defined in the class
        and returns the loss values for each epoch

      '''

      class attPlotsCallback(tf.keras.callbacks.Callback):

          def __init__(self, log_dir, test_data, problem_id):


              self.file_writer = tf.summary.create_file_writer(log_dir)
              self.test_data = test_data
              self.problem_id = problem_id
              self.images = None


          def plot_to_image(self, figure):

            """Converts the matplotlib plot specified by 'figure' to a PNG image and
            returns it. The supplied figure is closed and inaccessible after this call."""

            # Save the plot to a PNG in memory.
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            # Closing the figure prevents it from being displayed directly inside
            # the notebook.
            plt.close(figure)
            buf.seek(0)
            # Convert PNG buffer to TF image
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            # Add the batch dimension
            image = tf.expand_dims(image, 0)
            return image


          def on_epoch_end(self, epoch, logs=None):

              """Runs metrics and histogram summaries at epoch end."""
              data_point = None

              for batch in self.test_data:
                  data_point = batch[0][:1]
                  break

              _ , att_scores = self.model.layers[1](data_point)
              index = 0

              fig, ax = plt.subplots(att_scores.shape[0], att_scores.shape[2], figsize=(16,16))

              fig.figsize = (16*att_scores.shape[0], 16*att_scores.shape[2])

              if(self.problem_id == 1):

                  for i in range(att_scores.shape[0]):
                        for j in range(att_scores.shape[2]):
                          ax[i][j].matshow(att_scores[i][0][j], cmap='viridis', )

                          #ax[i][j].matshow(att_scores[i][0][j], cmap='viridis')
                          ax[i][j].set_title('epoch {} layer {}, head {}'.format(epoch, i, j))
                          ax[i][j].set_xticks(ticks = [i for i,_ in enumerate(data_point[0].numpy()) if _ == 3])
                          ax[i][j].set_yticks(ticks = [i for i,_ in enumerate(data_point[0].numpy()) if _ == 3])
                          ax[i][j].set_xticklabels([3,3])
                          ax[i][j].set_yticklabels([3,3])

                  if(self.images is not None):
                      self.images = tf.concat([self.images, self.plot_to_image(fig)], axis = 0)
                  else:
                      self.images = self.plot_to_image(fig)


                  print(self.images.shape)

                  with self.file_writer.as_default():
                    tf.summary.image("Training data", self.images, step=0)






      model = self.create_model()

      logdir = self.config['log_dir']
      tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir, histogram_freq = 1)
      att_plots = attPlotsCallback(logdir, self.test_dataset, self.problem_id)
      file_writer = tf.summary.create_file_writer(logdir)

      self.optimizer.learning_rate = self.learning_rate
      model.compile(optimizer= self.optimizer, loss=self.loss_function, metrics = self.metric)
      history = model.fit(self.train_dataset, epochs = self.epochs, validation_data = self.test_dataset, steps_per_epoch = len(self.train_dataset), callbacks=[tensorboard_callback,
                                                                                                                                              att_plots])

      return model, history



class Test():

    '''
      This class helps us in testing our model.

      The class helps in inferring a model and also plot the attention scores for our model

      Attributes:
        data_point: Test data points for which we want to test our model
        model: trained model object which is used for inferenece and plotting attention scores

    '''

    def __init__(self, data_point, model):

        ''' Initializes the variables '''

        self.model = model
        self.data_point = data_point

    def infer(self):

        ''' Gives prediction for the data points given by model '''

        return self.model(self.data_point)

    def attention_plots(self, layer=0, input_index=0):
        '''

          Plots heatmaps for our attention scores for a particular layer and input

          layer: layer number for which we want to plot attention scores
          input_index: input index for which we want to plot attention plots. In case of batch of inputs is sent

        '''


        _ , att_scores = self.model.layers[1](self.data_point)

        layer_att_scores = att_scores[layer]
        input_att_scores = layer_att_scores[input_index]
        num_heads = input_att_scores.shape[0]

        for head in range(num_heads):
          head_att_score = input_att_scores[head]
          fig = plt.figure(figsize=(16, 8))

          ax = fig.add_subplot(2, 4, head+1)
          #ax.matshow(head_att_score, cmap='viridis')
          sns.heatmap(head_att_score, cmap='PiYG')
          ax.set_xlabel('head {}'.format(head))

        plt.tight_layout()
        plt.show()







def _train(args):

    dataset_ob = dataset.DistanceDataset()
    train_dataset, test_dataset = dataset_ob.gen_data()

    train_ob = Train(args, train_dataset, test_dataset, 64)
    model, history = train_ob.train_model()


    # input_shape = (args.seq_length)
    # config = hyperparams.Config(args.num_heads, args.num_layers, args.emb_dim, args.seq_length, args.vocab_size, args.head_size, args.pos_embedding, args.agg_method, args.pos_embedding_type)
    # input = Input(input_shape)
    # op, att_scores = model._Model(config['emb_dim'], config['seq_length'], config['vocab_size'], config['pos_embedding'], config['num_heads'], config['head_size'], config['agg_method'],
    # config['pos_embedding_type'])(input)
    # output = Dense(1, activation='linear')(op)

    # _model = Model(inputs = input, outputs=output)
    # print(_model.summary())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_heads', type=int, default=1, help='Number of Attention Heads')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of Attention Layers')
    parser.add_argument('--emb_dim', type=int, default=128, help='Embedding Dimensions')
    parser.add_argument('--seq_length', type=int, default=512, help='input seq length')
    parser.add_argument('--vocab_size', type=int, default=2, help='length of vocab for input')
    parser.add_argument('--head_size', type=int, default=64, help='Size of the single dense layer head')
    parser.add_argument('--pos_embedding', type=bool, default=True, help='Weather to use positional embedding or not')
    parser.add_argument('--agg_method', type=str, default='TOKEN', help='Method used to feed into head. TOKEN Uses CLS token, SUM adds the output from attention')
    parser.add_argument('--pos_embedding_type', type=str, default='SIN_COS', help='Which type of positional encoding to use. SIN_COS for alternating sin cos, RANDOM for random pos encodings')
    parser.add_argument('--problem_id', type=int, default=1, help=''' Integer indicating which problem to solve?
    										Distance between 2 red tokens: 1
										Count number of red tokens: 2
										Find token that appears maximum time: 3
										Compute sequence length: 4
										Palindrome Sequence: 5
										Sorted Sequence: 6
										Sum: 7
										MAx: 8
										Min: 9
    										''')
    parser.add_argument('--optimizer', type=str, default='adam', help='Which optimizer to use for training')
    parser.add_argument('--loss_function', type=str, default='mean_squared_error', help='Which Loss Function to use for training')
    parser.add_argument('--metric', type=str, default='mean_squared_error', help='Which metric to use for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Which learning rate to use for training')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to run for training')
    args = parser.parse_args()

    _train(args)
