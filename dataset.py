import random
from random import sample
from tqdm.auto import tqdm
import pandas as pd
from collections import defaultdict
import numpy as np
import tensorflow as tf
import itertools
import math
import time
import hyperparms
import sys
import logging

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)

sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)


'''
Distance between 2 red tokens: 1
Count number of red tokens: 2
Find token that appears maximum time: 3
Compute sequence length: 4
Palindrome Sequence: 5
Sorted Sequence: 6
Sum: 7
MAx: 8
Min: 9

'''
class Dataset:

    '''
      Parent Class to all the data generation class. Holds all the methods and attributes used to generate data for our problems

      Attributes:

        agg_method: Used for problem 1 data. Refer to Config class for more details
        max_seq_length: maximum length of our input
        number_of_data_points: maximum number of datapoints to be generated in case where there are exponential order data points possible
    '''

    def __init__(self, vocab = None, agg_method = None, max_seq_length = 512, number_of_data_points = 100000):

        '''Initializes our variables'''

        self.agg_method = agg_method
        self.max_seq_length = max_seq_length
        self.number_of_data_points = 100000
        self.vocab = vocab

    def create_sequence_and_lables(self, indices):

        '''Method is used to generate data for problem 1 and problem 4'''


        sequence = None
        label = None

        if(self.agg_method == 'TOKEN'):
            sequence = tf.reduce_sum(tf.one_hot(indices, depth=self.max_seq_length, dtype='int64'), axis=0)
            #adding CLS token in our data
            sequence = sequence + 2
            sequence = tf.concat([tf.convert_to_tensor(np.ones((1)), dtype='int64'), sequence], axis=0)


            label = indices[1] - indices[0]

        elif(self.agg_method == 'SUM'):

            sequence = tf.reduce_sum(tf.one_hot(indices, depth=self.max_seq_length, dtype='int64'), axis=0)
            label = indices[1] - indices[0]

        else:
            sequence = tf.reduce_sum(tf.one_hot(tf.range(indices[0],indices[1]), depth=self.max_seq_length, dtype='int64'), axis=0)
            label = indices[1] - indices[0]

        return sequence, label

    def int_to_sequence(self, i):

      '''Method converts given integer into binary and returns the tensor form of it'''

      return tf.convert_to_tensor(list(map(int, (format(i.numpy(), 'b')))))

    def map_int_to_sequence(self, i):

      '''Method is used to generate data for problem 2'''

      sequence = tf.py_function(self.int_to_sequence, [i], tf.int32)
      label = tf.reduce_sum(sequence)
      # Adding 1 to differentiate between padding value and token ids
      return tf.add(sequence, 1), label



    def make_data(self, i):

        '''filling current i token in place of maximum # of 1 or 0 in the binary epresentation of i'''


        b = format(i,'b')
        num_one = len([1 for char in b if char == '1'])
        num_zero = len([1 for char in b if char == '0'])

        if(num_zero > num_one):
          point = [self.vocab[i%len(self.vocab)] if char == '0' else self.vocab[((self.max_seq_length - len(b)) + index)%len(self.vocab)] for index,char in enumerate(b)]

        else:
          point = [self.vocab[i%len(self.vocab)] if char == '1' else self.vocab[((self.max_seq_length - len(b)) + index)%len(self.vocab)] for index,char in enumerate(b)]


        return tf.convert_to_tensor(point,'string')


    def _gen_data(self, i):

        '''Creates data for problem 3'''

        sequence = tf.py_function(self.make_data, [i], 'string')
        label = tf.gather(self.vocab, i%len(self.vocab))
        return sequence,label



    def make_palindrome(self, i):

      '''
        This method is used for generating data for palindrome problem

        if i is even data point will be palindrome else non palindrome

      '''

      '''if len of binary representation of i is even then even lenght palindrome else odd length palindrome'''
      if(i%2 == 0):
          b = format(i,'b')
          l_b = len(b)
          b = '0'*(self.max_seq_length - len(b)) + b
          indices = [index%len(self.vocab) for index,char in enumerate(b) if char == '1']
          if(l_b %2 == 0):
            point = [self.vocab[i] for i in indices] + [self.vocab[i] for i in indices[: : -1]]

          else:
            point = [self.vocab[i] for i in indices] + [self.vocab[l_b]] + [self.vocab[i] for i in indices[: : -1]]
      else:
          b = format(i,'b')
          l_b = len(b)
          b = '0'*(self.max_seq_length - len(b)) + b
          indices = [index%len(self.vocab) for index,char in enumerate(b) if char == '1']
          point = [self.vocab[i] for i in indices] + [self.vocab[i] for i in [x*(x+4)%len(self.vocab) for x in indices]]

      return tf.convert_to_tensor(point,'string')

    def gen_data_pali(self, i):

        '''Generates data for palindrome data'''

        sequence = tf.py_function(self.make_palindrome, [i], 'string')

        if(i %2 == 0):
          label = 1
        else:
          label = 0
        return sequence,label



    def make_sort_data(self, j):

      ''' Method prepaes data for sorted data problem '''

      start = j
      b = format(start, 'b')
      s = len([1 for char in format(start, 'b') if char == '1'])
      m = 26
      num = [s]
      elem = j%67

      for i in range(elem):
        a = len([1 for char in format(start,'b') if char == '0'])
        s = len([1 for char in format(start, 'b') if char == '1'])

        c = s+a


        next = (a*num[-1] + c) % m

        if(j % 2 == 0):

          #preparing sort data
          if(next >= num[-1]):
            num.append(next)

            start = num[-1]
          else:
            start = next

        else:
            #prepraing unsort data
            num.append(next)
            start = num[-1]

      return tf.convert_to_tensor(list(map(lambda x: self.vocab[x] ,num)), 'string')

    def gen_sort_data(self, i):

        sequence = tf.py_function(self.make_sort_data, [i], 'string')


        if(i %2 == 0):
          label = 1
        else:
          label = 0

        return sequence,label


    def make_number_data(self, i):

      '''This method gives the datapoint as a tensor for problems 7,8 and 9'''

      return tf.convert_to_tensor([index%len(self.vocab) for index,char in enumerate('0'*(self.max_seq_length - len(format(i,'b'))) + format(i,'b')) if char == '1'],'int64')



    def gen_data_mms(self, i):

      sequence = tf.py_function(self.make_number_data, [i], 'int64')

      if(self.__class__.__name__ == 'SumDataset'):
        label = tf.reduce_sum(sequence)
      elif(self.__class__.__name__ == 'MaxDataset'):
        label = tf.reduce_max(sequence)
      else:
        label = tf.reduce_min(sequence)

      return sequence, label




class DistanceDataset(Dataset):

    '''
        This class generates and returns train and test data for distance problem (id: 1)

        Attributes:

        agg_method: Used for problem 1 data. Refer to Config class for more details
        max_seq_length: maximum length of our input
        number_of_data_points: maximum number of datapoints to be generated in case where there are exponential order data points possible

    '''

    def __init__(self, agg_method = 'TOKEN', max_seq_length = 512, number_data_points = 100000):

        '''Initializes our variables'''

        Dataset.__init__(self, agg_method = agg_method, max_seq_length = max_seq_length, number_of_data_points = number_data_points)


        self.max_seq_length = max_seq_length
        self.number_data_points = number_data_points
        #self.config = hyperparms.Config()


    def gen_data(self):

        '''This method generates the data for our problem'''

        #logger.info('**************GENERATING DATA FOR PROBLEM 1 *************\n')

        indices = [[i, j] for i in range(self.max_seq_length) for j in range(i+1, self.max_seq_length)]

        #sequence, label = self.create_sequence_and_lables(indices, 'TOKEN', self.max_seq_length)

        dataset = tf.data.Dataset.from_tensor_slices(indices)
        dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=False)
        dataset = dataset.map(self.create_sequence_and_lables)


        train_size = int(0.8 * len(indices))
        train_dataset, test_dataset = dataset.take(train_size), dataset.skip(train_size)


        train_dataset = train_dataset.batch(64, drop_remainder = True)
        test_dataset = test_dataset.batch(64, drop_remainder = True)

        return train_dataset, test_dataset


class CountRedTokenDataset(Dataset):

    '''
        This class generates and returns train and test data for Counting Tokens problem (id: 2)

        Attributes:

        max_seq_length: maximum length of our input
        number_of_data_points: maximum number of datapoints to be generated in case where there are exponential order data points possible

    '''


    def __init__(self , max_seq_length = 512, number_data_points = 100000):

        '''Initilaizes the variables'''

        Dataset.__init__(self)

        self.max_seq_length = max_seq_length
        self.number_data_points = number_data_points

    def gen_data(self):

        '''This method generates the data for our problem'''

        #logger.info('**************GENERATING DATA FOR PROBLEM 2 *************\n')

        dataset = tf.data.Dataset.from_tensor_slices(tf.range(self.number_data_points))
        dataset = dataset.map(self.map_int_to_sequence, num_parallel_calls=tf.data.AUTOTUNE)

        train_size = int(0.8 * self.number_data_points)
        train_dataset, test_dataset = dataset.take(train_size), dataset.skip(train_size)

        train_dataset = train_dataset.padded_batch(512, padded_shapes=((None,), ()), drop_remainder=True)
        test_dataset = test_dataset.padded_batch(512, padded_shapes=((None,), ()), drop_remainder = True)


        return train_dataset, test_dataset


class MaxTokenDataset(Dataset):

    '''
        This class generates and returns train and test data for Maximum number Tokens problem (id: 3)

        Attributes:

        vocab: Vocabulary to be used to generate the data to be given in form of a list
        max_seq_length: maximum length of our input
        number_of_data_points: maximum number of datapoints to be generated in case where there are exponential order data points possible

    '''

    def __init__(self , vocab, max_seq_length = 512, number_data_points = 100000):

        '''Initilaizes the variables'''

        Dataset.__init__(self, vocab)

        self.max_seq_length = max_seq_length
        self.number_data_points = number_data_points

    def gen_data(self):

      '''generates data for problem 3'''

      #logger.info('**************GENERATING DATA FOR PROBLEM 3 *************\n')

      dataset = tf.data.Dataset.from_tensor_slices(tf.range(self.number_data_points))

      dataset = dataset.map(self._gen_data, num_parallel_calls=tf.data.AUTOTUNE)
      dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=False)


      train_size = int(0.8 * self.number_data_points)
      train_dataset, test_dataset = dataset.take(train_size), dataset.skip(train_size)

      train_dataset = train_dataset.padded_batch(512, padded_shapes=((None,), ()), drop_remainder= True)
      test_dataset = test_dataset.padded_batch(512, padded_shapes=((None,), ()), drop_remainder=True)


      return train_dataset, test_dataset


class SeqLenDataset(Dataset):

    '''
        This class generates and returns train and test data for Sequence length  problem (id: 4)

        Attributes:

        max_seq_length: maximum length of our input
        number_of_data_points: maximum number of datapoints to be generated in case where there are exponential order data points possible

    '''

    def __init__(self, max_seq_length = 512, number_data_points = 100000):

        '''Initializes our variables'''

        Dataset.__init__(self, agg_method = None)


        self.max_seq_length = max_seq_length
        self.number_data_points = number_data_points


    def gen_data(self):

      '''generates data for problem 3'''

      #logger.info('**************GENERATING DATA FOR PROBLEM 4 *************\n')

      indices = [[0, j] for j in range(1, self.max_seq_length)]

      dataset = tf.data.Dataset.from_tensor_slices(indices)
      dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=False)
      dataset = dataset.map(self.create_sequence_and_lables)


      train_size = int(0.8 * len(indices))
      train_dataset, test_dataset = dataset.take(train_size), dataset.skip(train_size)


      train_dataset = train_dataset.batch(64, drop_remainder=True)
      test_dataset = test_dataset.batch(64, drop_remainder = True)

      return train_dataset, test_dataset


class PalindromeDataset(Dataset):

    '''
        This class generates and returns train and test data for  Palindrome  problem (id: 5)

        Attributes:

        vocab: vocab used for our data , given as a list
        max_seq_length: maximum length of our input
        number_of_data_points: maximum number of datapoints to be generated in case where there are exponential order data points possible

    '''



    def __init__(self, vocab, max_seq_length = 512, number_data_points = 100000):

        '''Initializes our variables'''

        Dataset.__init__(self, vocab)


        self.max_seq_length = max_seq_length
        self.number_data_points = number_data_points



    def gen_data(self):

        '''Generates data'''

        #logger.info('**************GENERATING DATA FOR PROBLEM 5 *************\n')


        dataset = tf.data.Dataset.from_tensor_slices(tf.range(1,self.number_data_points))
        dataset = dataset.map(self.gen_data_pali, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=False)

        train_size = int(0.8 * self.number_data_points)
        train_dataset, test_dataset = dataset.take(train_size), dataset.skip(train_size)

        train_dataset = train_dataset.padded_batch(512, padded_shapes=((None,), ()), drop_remainder=True)
        test_dataset = test_dataset.padded_batch(512, padded_shapes=((None,), ()), drop_remainder=True)


        return train_dataset, test_dataset


class SortedDataset(Dataset):

    '''
        This class generates and returns train and test data for  Sorted Data  problem (id: 6)

        Attributes:

        vocab: vocab used for our data , given as a list
        max_seq_length: maximum length of our input
        number_of_data_points: maximum number of datapoints to be generated in case where there are exponential order data points possible

    '''




    def __init__(self, vocab, max_seq_length = 512, number_data_points = 100000):

        '''Initializes our variables'''

        Dataset.__init__(self, vocab)


        self.max_seq_length = max_seq_length
        self.number_data_points = number_data_points


    def gen_data(self):

          '''Generates data'''

          #logger.info('**************GENERATING DATA FOR PROBLEM 6 *************\n')


          dataset = tf.data.Dataset.from_tensor_slices(tf.range(self.number_data_points))

          dataset = dataset.map(self.gen_sort_data, num_parallel_calls=tf.data.AUTOTUNE)

          dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=False)

          train_size = int(0.8 * self.number_data_points)
          train_dataset, test_dataset = dataset.take(train_size), dataset.skip(train_size)

          train_dataset = train_dataset.padded_batch(512, padded_shapes=((None,), ()), drop_remainder=True)
          test_dataset = test_dataset.padded_batch(512, padded_shapes=((None,), ()), drop_remainder=True)


          return train_dataset, test_dataset


class SumDataset(Dataset):

    '''
        This class generates and returns train and test data for Sum data  problem (id: 7)

        Attributes:

        vocab: vocab used for our data , given as a list
        max_seq_length: maximum length of our input
        number_of_data_points: maximum number of datapoints to be generated in case where there are exponential order data points possible

    '''

    def __init__(self, vocab, max_seq_length = 512, number_data_points = 100000):

        '''Initializes our variables'''

        Dataset.__init__(self, vocab)


        self.max_seq_length = max_seq_length
        self.number_data_points = number_data_points


    def gen_data(self):

        dataset = tf.data.Dataset.from_tensor_slices(tf.range(2**27,2**27 + self.number_data_points))
        dataset = dataset.map(self.gen_data_mms, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=False)

        train_size = int(0.8 * self.number_data_points)
        train_dataset, test_dataset = dataset.take(train_size), dataset.skip(train_size)

        train_dataset = train_dataset.padded_batch(512, padded_shapes=((None,), ()), drop_remainder=True)
        test_dataset = test_dataset.padded_batch(512, padded_shapes=((None,), ()), drop_remainder = True)


        return train_dataset, test_dataset


class MaxDataset(Dataset):

    '''
        This class generates and returns train and test data for Max data  problem (id: 8)

        Attributes:

        vocab: vocab used for our data , given as a list
        max_seq_length: maximum length of our input
        number_of_data_points: maximum number of datapoints to be generated in case where there are exponential order data points possible

    '''

    def __init__(self, vocab, max_seq_length = 512, number_data_points = 100000):

        '''Initializes our variables'''

        Dataset.__init__(self, vocab)


        self.max_seq_length = max_seq_length
        self.number_data_points = number_data_points


    def gen_data(self):

        dataset = tf.data.Dataset.from_tensor_slices(tf.range(2**27,2**27 + self.number_data_points))
        dataset = dataset.map(self.gen_data_mms, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=False)

        train_size = int(0.8 * self.number_data_points)
        train_dataset, test_dataset = dataset.take(train_size), dataset.skip(train_size)

        train_dataset = train_dataset.padded_batch(512, padded_shapes=((None,), ()), drop_remainder=True)
        test_dataset = test_dataset.padded_batch(512, padded_shapes=((None,), ()), drop_remainder= True)


        return train_dataset, test_dataset



class MinDataset(Dataset):

    '''
        This class generates and returns train and test data for Min data  problem (id: 9)

        Attributes:

        vocab: vocab used for our data , given as a list
        max_seq_length: maximum length of our input
        number_of_data_points: maximum number of datapoints to be generated in case where there are exponential order data points possible

    '''

    def __init__(self, vocab, max_seq_length = 512, number_data_points = 100000):

        '''Initializes our variables'''

        Dataset.__init__(self, vocab)


        self.max_seq_length = max_seq_length
        self.number_data_points = number_data_points


    def gen_data(self):

        dataset = tf.data.Dataset.from_tensor_slices(tf.range(2**27,2**27 + self.number_data_points))
        dataset = dataset.map(self.gen_data_mms, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=False)

        train_size = int(0.8 * self.number_data_points)
        train_dataset, test_dataset = dataset.take(train_size), dataset.skip(train_size)

        train_dataset = train_dataset.padded_batch(512, padded_shapes=((None,), ()), drop_remainder=True, padding_values = tf.constant(100, dtype='int64'))
        test_dataset = test_dataset.padded_batch(512, padded_shapes=((None,), ()), drop_remainder= True, padding_values = tf.constant(100, dtype='int64'))


        return train_dataset, test_dataset
