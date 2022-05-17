import unittest
import random
import dataset
import tensorflow as tf
from collections import defaultdict
import logging
import sys
from tqdm import tqdm
import model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)

sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)






class testDistanceDataset(unittest.TestCase):


    '''
      Test case used to test the Distance Dataset

    '''

    def setUp(self):
        self.train_dataset_token, self.test_dataset_token = dataset.DistanceDataset(agg_method = 'TOKEN').gen_data()
        self.train_dataset_sum, self.test_dataset_sum = dataset.DistanceDataset(agg_method = 'SUM').gen_data()
        self.batch_size = 64

    def test_Dtype(self):

        '''Test the data type of train and test data'''

        logger.info('**************** Testing DistanceDataset data type ****************************')

        train_batch = None
        test_batch = None

        for b in self.train_dataset_token:
            train_batch = b
            break

        for b in self.test_dataset_token:
            test_batch = b
            break

        self.assertEqual(train_batch[0].dtype, tf.int64, 'Data Type has to be int64 or int32')
        self.assertEqual(train_batch[1].dtype, tf.int32, 'Data Type has to be int64 or int32')
        self.assertEqual(test_batch[0].dtype, tf.int64, 'Data Type has to be int64 or int32')
        self.assertEqual(test_batch[1].dtype, tf.int32, 'Data Type has to be int64 or int32')

    def test_Shape(self):

        ''''Test the shape of data for different aggregation method'''

        logger.info('**************** Testing DistanceDataset shape ****************************')

        token_batch = None
        sum_batch = None

        for b in self.train_dataset_token:
            token_batch = b
            break

        for b in self.train_dataset_sum:
            sum_batch = b
            break

        self.assertEqual(token_batch[0].shape, tf.TensorShape([self.batch_size,513]), 'Invalid data Shapes')
        self.assertEqual(sum_batch[0].shape, tf.TensorShape([self.batch_size,512]), 'Invalid data Shapes')

    def test_Valididty(self):

        '''Test the validity of rnadomly selected 50 Points from train data'''

        logger.info('**************** Testing DistanceDataset Validity ****************************')

        train_sample = []
        train_sample_sum = []
        number_samples = 0

        #picking 50 points from train data
        for batch in tqdm(self.train_dataset_token):
            if(number_samples < 50):
              i = random.randint(0, self.batch_size-1)
              train_sample.append((batch[0][i], batch[1][i]))
              number_samples += 1


        number_samples = 0
        #picking 50 points from train data
        for batch in tqdm(self.train_dataset_sum):
            if(number_samples < 50):
              i = random.randint(0, self.batch_size-1)
              train_sample_sum.append((batch[0][i], batch[1][i]))
              number_samples += 1



        '''Checking is each data point has 2 target tokens and target value is less than length of sequence'''
        for dp,target in tqdm(train_sample):
            t = list(filter(lambda x: dp[x] == 3, list(range(513))))
            self.assertEqual(len(t), 2, 'Invalid Data Point')
            self.assertTrue(target < 513 , 'Invalid Target Value')
            self.assertEqual((t[1] - t[0]), target, 'Invalid Data Point(Target not matching)')


        '''Checking is each data point has 2 target tokens and target value is less than length of sequence'''
        for dp,target in tqdm(train_sample_sum):
            t = list(filter(lambda x: dp[x] == 1, list(range(512))))
            self.assertEqual(len(t), 2, 'Invalid Data Point')
            self.assertTrue(target < 512 , 'Invalid Target Value')
            self.assertEqual((t[1] - t[0]), target, 'Invalid Data Point(Target not matching)')




class testRedTokenDataset(unittest.TestCase):

    def setUp(self):
        self.train_dataset, self.test_dataset = dataset.CountRedTokenDataset().gen_data()
        self.batch_size = 512


    def test_Dtype(self):

        '''Test the data type of train and test data'''

        logger.info('**************** Testing RedTokenDataset data type ****************************')

        train_batch = None
        test_batch = None

        for b in self.train_dataset:
            train_batch = b
            break

        for b in self.test_dataset:
            test_batch = b
            break

        self.assertEqual(train_batch[0].dtype, tf.int32, 'Data Type has to be int64 or int32')
        self.assertEqual(test_batch[0].dtype, tf.int32, 'Data Type has to be int64 or int32')


    def test_Validity(self):

        '''Test the validity of rnadomly selected 50 Points from train data'''

        logger.info('**************** Testing RedTokenDataset Validity ****************************')

        train_sample = []
        samples = 0

        for batch in tqdm(self.train_dataset):
            if(samples < 50):
                i = random.randint(0, self.batch_size-1)
                train_sample.append((batch[0][i], batch[1][i]))
                samples += 1

        for dp, target in tqdm(train_sample):
            self.assertEqual(len(list(filter(lambda x: x==2, dp))), target, 'Invalid Data point')




class testMaxTokenDataset(unittest.TestCase):

    def setUp(self):
        self.vocab = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        self.train_dataset, self.test_dataset = dataset.MaxTokenDataset(self.vocab).gen_data()
        self.batch_size = 512


    def test_validity(self):

        '''Test the validity of rnadomly selected 50 Points from train data'''

        logger.info('**************** Testing MaxTokenDataset Validity ****************************')

        train_sample = []
        samples = 0

        for batch in tqdm(self.train_dataset):
            if(samples < 50):
                i = random.randint(0, self.batch_size-1)
                train_sample.append((batch[0][i], batch[1][i]))
                samples += 1

        for dp, target in tqdm(train_sample):
            max_token = None
            di = defaultdict(int)
            for val in dp:
                if(val.numpy() != b''):
                    di[val.numpy()] += 1
            max_token = max(di, key= lambda x: di[x])

            self.assertEqual(max_token, target.numpy(), 'Invalid Data Point')


class testSeqLenDataset(unittest.TestCase):

    def setUp(self):
        self.train_dataset, self.test_dataset = dataset.SeqLenDataset().gen_data()
        self.batch_size = 64


    def test_Dtype(self):

        '''Test the data type of train and test data'''

        logger.info('**************** Testing SeqLenDataset data type ****************************')

        train_batch = None
        test_batch = None

        for b in self.train_dataset:
            train_batch = b
            break

        for b in self.test_dataset:
            test_batch = b
            break

        self.assertEqual(train_batch[0].dtype, tf.int64, 'Data Type has to be int64 or int32')
        self.assertEqual(train_batch[1].dtype, tf.int32, 'Data Type has to be int64 or int32')
        self.assertEqual(test_batch[0].dtype, tf.int64, 'Data Type has to be int64 or int32')
        self.assertEqual(test_batch[1].dtype, tf.int32, 'Data Type has to be int64 or int32')

    def test_Shape(self):

        ''''Test the shape of data for different aggregation method'''

        logger.info('**************** Testing SeqLenDataset shape ****************************')

        batch = None

        for b in self.train_dataset:
            batch = b
            break


        self.assertEqual(batch[0].shape, tf.TensorShape([self.batch_size,512]), 'Invalid data Shapes')



    def test_Validity(self):

        '''Test the validity of rnadomly selected 50 Points from train data'''

        logger.info('**************** Testing SeqLenDataset validity ****************************')

        train_sample = []
        samples = 0

        for batch in tqdm(self.train_dataset):
            if(samples < 50):
                i = random.randint(0, self.batch_size-1)
                train_sample.append((batch[0][i], batch[1][i]))
                samples += 1

        for dp, target in tqdm(train_sample):
            self.assertEqual(len(list(filter(lambda x: x==1, dp))), target, 'Invalid Datapoint')



class testPalindromeDataset(unittest.TestCase):

    def setUp(self):
        self.vocab = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        self.train_dataset, self.test_dataset = dataset.PalindromeDataset(self.vocab).gen_data()
        self.batch_size = 512


    def test_validity(self):

        '''Test the validity of rnadomly selected 50 Points from train data'''

        logger.info('**************** Testing PalindromeDataset Validity ****************************')

        train_sample = []
        test_sample = []
        samples = 0

        for batch in tqdm(self.train_dataset):
            if(samples < 50):
                i = random.randint(0, self.batch_size-1)
                train_sample.append((batch[0][i], batch[1][i]))
                samples += 1

        samples = 0
        for batch in tqdm(self.test_dataset):
            if(samples < 50):
                i = random.randint(0, self.batch_size-1)
                test_sample.append((batch[0][i], batch[1][i]))
                samples += 1


        for dp, target in tqdm(train_sample):
            filter_dp = []
            for val in dp:
                if(val.numpy() != b''):
                    filter_dp.append(val.numpy())

            if(filter_dp == filter_dp[::-1]):
                self.assertEqual(1, target, 'Invalid train Data Point')
            else:
                self.assertEqual(0, target, 'Invalid train Data Point')


        for dp, target in tqdm(test_sample):
            filter_dp = []
            for val in dp:
                if(val.numpy() != b''):
                    filter_dp.append(val.numpy())

            if(filter_dp == filter_dp[::-1]):
                self.assertEqual(1, target, 'Invalid test Data Point')
            else:
                self.assertEqual(0, target, 'Invalid test Data Point')


class testSortedDataset(unittest.TestCase):

    def setUp(self):
        print('here')
        self.vocab = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        self.train_dataset, self.test_dataset = dataset.SortedDataset(self.vocab).gen_data()
        self.batch_size = 512


    def test_validity(self):

        '''Test the validity of rnadomly selected 50 Points from train data'''



        logger.info('**************** Testing sortedDataset validity ****************************')

        train_sample = []
        samples = 0

        for batch in (self.train_dataset):
            for i in tqdm(range(50)):
                train_sample.append((batch[0][i], batch[1][i]))
                samples += 1
            break

        for dp, target in tqdm(train_sample):
            filter_dp = [val.numpy() for val in dp if val.numpy() != b'']

            if(sorted(filter_dp) == filter_dp):
                self.assertEqual(target, 1, 'Invalid Data Point')
            else:
                self.assertEqual(target, 0, 'Invalid Data Point')


class testSumDataset(unittest.TestCase):

    def setUp(self):
        self.vocab = list(range(10))
        self.train_dataset, self.test_dataset = dataset.SumDataset(self.vocab).gen_data()
        self.batch_size = 512


    def test_Dtype(self):

        '''Test the data type of train and test data'''

        logger.info('**************** Testing SumDataset data type ****************************')

        train_batch = None
        test_batch = None

        for b in self.train_dataset:
            train_batch = b
            break

        for b in self.test_dataset:
            test_batch = b
            break

        self.assertEqual(train_batch[0].dtype, tf.int64, 'Data Type has to be int64 or int32')
        self.assertEqual(train_batch[1].dtype, tf.int64, 'Data Type has to be int64 or int32')
        self.assertEqual(test_batch[0].dtype, tf.int64, 'Data Type has to be int64 or int32')
        self.assertEqual(test_batch[1].dtype, tf.int64, 'Data Type has to be int64 or int32')


    def test_Validity(self):

        '''Test the validity of rnadomly selected 50 Points from train data'''

        logger.info('**************** Testing SumDataset validity ****************************')

        train_sample = []
        samples = 0

        for batch in tqdm(self.train_dataset):
            if(samples < 50):
                i = random.randint(0, self.batch_size-1)
                train_sample.append((batch[0][i], batch[1][i]))
                samples += 1

        for dp, target in tqdm(train_sample):
            self.assertEqual(sum(dp), target, 'Invalid Datapoint')



class testMaxDataset(unittest.TestCase):

    def setUp(self):
        self.vocab = list(range(10))
        self.train_dataset, self.test_dataset = dataset.MaxDataset(self.vocab).gen_data()
        self.batch_size = 512


    def test_Dtype(self):

        '''Test the data type of train and test data'''

        logger.info('**************** Testing MaxDataset data type ****************************')

        train_batch = None
        test_batch = None

        for b in self.train_dataset:
            train_batch = b
            break

        for b in self.test_dataset:
            test_batch = b
            break

        self.assertEqual(train_batch[0].dtype, tf.int64, 'Data Type has to be int64 or int32')
        self.assertEqual(train_batch[1].dtype, tf.int64, 'Data Type has to be int64 or int32')
        self.assertEqual(test_batch[0].dtype, tf.int64, 'Data Type has to be int64 or int32')
        self.assertEqual(test_batch[1].dtype, tf.int64, 'Data Type has to be int64 or int32')


    def test_Validity(self):

        '''Test the validity of rnadomly selected 50 Points from train data'''

        logger.info('**************** Testing MaxDataset validity ****************************')

        train_sample = []
        samples = 0

        for batch in tqdm(self.train_dataset):
            if(samples < 50):
                i = random.randint(0, self.batch_size-1)
                train_sample.append((batch[0][i], batch[1][i]))
                samples += 1

        for dp, target in tqdm(train_sample):
            self.assertEqual(max(dp), target, 'Invalid Datapoint')



class testMinDataset(unittest.TestCase):

    def setUp(self):
        self.vocab = list(range(10))
        self.train_dataset, self.test_dataset = dataset.MinDataset(self.vocab).gen_data()
        self.batch_size = 512


    def test_Dtype(self):

        '''Test the data type of train and test data'''

        logger.info('**************** Testing MinDataset data type ****************************')

        train_batch = None
        test_batch = None

        for b in self.train_dataset:
            train_batch = b
            break

        for b in self.test_dataset:
            test_batch = b
            break

        self.assertEqual(train_batch[0].dtype, tf.int64, 'Data Type has to be int64 or int32')
        self.assertEqual(train_batch[1].dtype, tf.int64, 'Data Type has to be int64 or int32')
        self.assertEqual(test_batch[0].dtype, tf.int64, 'Data Type has to be int64 or int32')
        self.assertEqual(test_batch[1].dtype, tf.int64, 'Data Type has to be int64 or int32')


    def test_Validity(self):

        '''Test the validity of rnadomly selected 50 Points from train data'''

        logger.info('**************** Testing MinDataset validity ****************************')

        train_sample = []
        samples = 0

        for batch in tqdm(self.train_dataset):
            if(samples < 50):
                i = random.randint(0, self.batch_size-1)
                train_sample.append((batch[0][i], batch[1][i]))
                samples += 1

        for dp, target in tqdm(train_sample):
            self.assertEqual(min(dp), target, 'Invalid Datapoint')



class testModel(unittest.TestCase):

    '''
        TEst the model

    '''

    def setUp(self):
        input = Input(513, dtype='int64')
        op, att_scores = model._Model(emb_dim = 64, seq_length = 512, vocab_size = 2, pos_embedding= True, num_heads = 4, head_size = 32, agg_method = 'SUM', embedding_type = 'SIN_COS', num_att_layers = 3)(input)
        output = Dense(1, activation='linear')(op)

        att_model = Model(inputs = input, outputs=output)

        self.model = att_model

        self.input = tf.random.uniform(shape=(64,513), minval=1, maxval=3, dtype='int64')



    def test_model(self):

        '''
            Test the shapes of model
        '''
        output,att_scores = self.model.layers[1](self.input)
        self.assertEqual(output.shape, (64,32), 'Invalid attention layers Output')
        self.assertEqual(att_scores.shape, (3,64,4,513,513), 'Invalid Attention Scores Output')
        output = self.model.layers[-1](output)
        self.assertEqual(output.shape, (64,1))


if __name__ == '__main__':
    unittest.main()
