from imports.py import *



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


def Generate_data(vocab , problem_id , max_seq_length = 512, number_data_points = 100000):

  '''This function generates the data based on our inputs

  vocab: vaocabulary of chars we are working with
  max_seq_length: max sequence length our single input can be of
  number_data_points: number of data points we want to generate
  problem_id: id of the problem for which data has to be generated'''

  data = {}

  #Generating for problem 1
  if(problem_id == 1):
    
    print('**************GENERATING DATA FOR PROBLEM 1 *************\n')

    indices = [[i, j] for i in range(max_seq_length) for j in range(i+1, max_seq_length)]

    def create_sequence_and_lables(indices):
        sequence = tf.reduce_sum(tf.one_hot(indices, depth=max_seq_length, dtype='int64'), axis=0)
        label = indices[1] - indices[0]
        return sequence, label

    dataset = tf.data.Dataset.from_tensor_slices(indices)
    dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=False)
    dataset = dataset.map(create_sequence_and_lables)


    train_size = int(0.8 * len(indices))
    train_dataset, test_dataset = dataset.take(train_size), dataset.skip(train_size)


    train_dataset = train_dataset.batch(64, drop_remainder = True)
    test_dataset = test_dataset.batch(64, drop_remainder = True)

    return train_dataset, test_dataset

  #generating for problem 2
  if(problem_id == 2):
    print('**************GENERATING DATA FOR PROBLEM 2 *************\n')


    def int_to_sequence(i):
      return tf.convert_to_tensor(list(map(int, (format(i.numpy(), 'b')))))

    def map_int_to_sequence(i):
      sequence = tf.py_function(int_to_sequence, [i], tf.int32)
      label = tf.reduce_sum(sequence)
      # Adding 1 to differentiate between padding value and token ids
      return tf.add(sequence, 1), label

    dataset = tf.data.Dataset.from_tensor_slices(tf.range(100000))
    dataset = dataset.map(map_int_to_sequence, num_parallel_calls=tf.data.AUTOTUNE)

    train_size = int(0.8 * number_data_points)
    train_dataset, test_dataset = dataset.take(train_size), dataset.skip(train_size)

    train_dataset = train_dataset.padded_batch(512, padded_shapes=((None,), ()))
    test_dataset = test_dataset.padded_batch(512, padded_shapes=((None,), ()))


    return train_dataset, test_dataset



  '''data for problem 3'''
  if(problem_id == 3):

      print('**************GENERATING DATA FOR PROBLEM 3 *************\n')

      '''filling current i token in place of maximum # of 1 or 0 in the binary epresentation of i'''
    
      def make_data(i):
        
        b = format(i,'b')
        num_one = len([1 for char in b if char == '1'])
        num_zero = len([1 for char in b if char == '0'])

        if(num_zero > num_one):
          point = [vocab[i%len(vocab)] if char == '0' else vocab[((max_seq_length - len(b)) + index)%len(vocab)] for index,char in enumerate(b)]

        else:
          point = [vocab[i%len(vocab)] if char == '1' else vocab[((max_seq_length - len(b)) + index)%len(vocab)] for index,char in enumerate(b)]


        return tf.convert_to_tensor(point,'string')
        

      def gen_data(i):
        sequence = tf.py_function(make_data, [i], 'string')
        label = tf.gather(vocab, i%len(vocab))
        return sequence,label

      dataset = tf.data.Dataset.from_tensor_slices(tf.range(number_data_points))

      dataset = dataset.map(gen_data, num_parallel_calls=tf.data.AUTOTUNE)
      dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=False)


      train_size = int(0.8 * number_data_points)
      train_dataset, test_dataset = dataset.take(train_size), dataset.skip(train_size)

      train_dataset = train_dataset.padded_batch(512, padded_shapes=((None,), ()))
      test_dataset = test_dataset.padded_batch(512, padded_shapes=((None,), ()))


      return train_dataset, test_dataset

  '''Data for problem 4'''
  if(problem_id == 4):

      print('**************GENERATING DATA FOR PROBLEM 4 *************\n')

      indices = [[0, j] for j in range(1, max_seq_length)]

      def create_sequence_and_lables(indices):
        sequence = tf.reduce_sum(tf.one_hot(tf.range(indices[0],indices[1]), depth=max_seq_length), axis=0)
        label = indices[1] - indices[0]
        return sequence, label

      dataset = tf.data.Dataset.from_tensor_slices(indices)
      dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=False)
      dataset = dataset.map(create_sequence_and_lables)


      train_size = int(0.8 * len(indices))
      train_dataset, test_dataset = dataset.take(train_size), dataset.skip(train_size)


      train_dataset = train_dataset.batch(64)
      test_dataset = test_dataset.batch(64)

      return train_dataset, test_dataset

  '''data for problem 5'''
  if(problem_id == 5):

    print('**************GENERATING DATA FOR PROBLEM 5 *************\n')


    '''if i is even data point will be palindrome else non palindrome'''
    def make_palindrome(i):
      '''if len of binary representation of i is even then even lenght palindrome else odd length palindrome'''
      if(i%2 == 0):
          b = format(i,'b')
          l_b = len(b)
          b = '0'*(max_seq_length - len(b)) + b
          indices = [index%len(vocab) for index,char in enumerate(b) if char == '1']
          if(l_b %2 == 0):
            point = [vocab[i] for i in indices] + [vocab[i] for i in indices[: : -1]]

          else:
            point = [vocab[i] for i in indices] + [vocab[l_b]] + [vocab[i] for i in indices[: : -1]]
      else:
          b = format(i,'b')
          l_b = len(b)
          b = '0'*(max_seq_length - len(b)) + b
          indices = [index%len(vocab) for index,char in enumerate(b) if char == '1']
          point = [vocab[i] for i in indices] + [vocab[i] for i in [x*(x+4)%len(vocab) for x in indices]]

      return tf.convert_to_tensor(point,'string')

    def gen_data(i):
        sequence = tf.py_function(make_palindrome, [i], 'string')

        if(i %2 == 0):
          label = 1
        else:
          label = 0
        return sequence,label

    dataset = tf.data.Dataset.from_tensor_slices(tf.range(1,number_data_points))
    dataset = dataset.map(gen_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=False)

    train_size = int(0.8 * number_data_points)
    train_dataset, test_dataset = dataset.take(train_size), dataset.skip(train_size)

    train_dataset = train_dataset.padded_batch(512, padded_shapes=((None,), ()))
    test_dataset = test_dataset.padded_batch(512, padded_shapes=((None,), ()))


    return train_dataset, test_dataset



  if(problem_id == 6):

    print('**************GENERATING DATA FOR PROBLEM 6 *************\n')


    def make_sort_data(j):

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

      return tf.convert_to_tensor(list(map(lambda x: vocab[x] ,num)), 'string')

    def gen_sort_data(i):

        sequence = tf.py_function(make_sort_data, [i], 'string')


        if(i %2 == 0):
          label = 1
        else:
          label = 0

        return sequence,label

    
    dataset = tf.data.Dataset.from_tensor_slices(tf.range(number_data_points))

    dataset = dataset.map(gen_sort_data, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=False)

    train_size = int(0.8 * number_data_points)
    train_dataset, test_dataset = dataset.take(train_size), dataset.skip(train_size)

    train_dataset = train_dataset.padded_batch(512, padded_shapes=((None,), ()))
    test_dataset = test_dataset.padded_batch(512, padded_shapes=((None,), ()))


    return train_dataset, test_dataset



  if(problem_id == 7 or problem_id == 8 or problem_id == 9):

    print('**************GENERATING DATA FOR PROBLEM 7/8/9 *************\n')


    def make_number_data(i):
      return tf.convert_to_tensor([index%len(vocab) for index,char in enumerate('0'*(max_seq_length - len(format(i,'b'))) + format(i,'b')) if char == '1'],'int64')

    def gen_data(i):

      sequence = tf.py_function(make_number_data, [i], 'int64')

      if(problem_id == 7):
        label = tf.reduce_sum(sequence)
      elif(problem_id == 8):
        label = tf.reduce_max(sequence)
      else:
        label = tf.reduce_min(sequence)

      return sequence, label

    dataset = tf.data.Dataset.from_tensor_slices(tf.range(2**27,2**27 + number_data_points))
    dataset = dataset.map(gen_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=False)

    train_size = int(0.8 * number_data_points)
    train_dataset, test_dataset = dataset.take(train_size), dataset.skip(train_size)

    train_dataset = train_dataset.padded_batch(512, padded_shapes=((None,), ()))
    test_dataset = test_dataset.padded_batch(512, padded_shapes=((None,), ()))


    return train_dataset, test_dataset
