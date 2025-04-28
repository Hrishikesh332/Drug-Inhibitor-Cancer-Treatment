from keras.layers import Dense, BatchNormalization, Activation, Dropout
from src import setting
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

class DrugsCombModel():

    def __init__(self, drug_a_features_len, drug_b_features_len, cl_features_len):

        self.drug_a_features_len = drug_a_features_len
        self.drug_b_features_len = drug_b_features_len
        self.cl_features_len = cl_features_len
        self.input_len = int(drug_a_features_len) + int(drug_b_features_len) + int(cl_features_len)

    def __mlp_model(self, nodes_nums, input_len):

        from keras.models import Sequential
        model = Sequential()

        for i in range(len(nodes_nums)):

            if i == 0:
                model.add(Dense(nodes_nums[i], input_shape=(input_len,)))
            else:
                model.add(Dense(nodes_nums[i]))

            model.add(BatchNormalization())
            model.add(Activation(setting.activation_method[i % len(setting.activation_method)]))
            model.add(Dropout(rate=setting.dropout[i % len(setting.dropout)]))

        model.add(Dense(1))

        return model

    def correlation_coefficient_loss(y_true, y_pred):
        x = y_true
        y = y_pred
        mx = K.mean(x)
        my = K.mean(y)
        xm, ym = x - mx, y - my
        r_num = K.sum(tf.multiply(xm, ym))
        r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
        r = r_num / r_den

        r = K.maximum(K.minimum(r, 1.0), -1.0)
        return 1 - K.square(r)

    @classmethod
    def get_loss(cls):

        # select the loss function
        if setting.loss == 'pearson_correlation':
            my_loss = cls.correlation_coefficient_loss

        else:
            my_loss = 'mse'

        return my_loss

    @classmethod
    def compile_transfer_learning_model(cls, model):

        custimized_rmsprop = Adam(lr=setting.start_lr, decay=setting.lr_decay)
        model.compile(optimizer=custimized_rmsprop, loss=cls.get_loss(), metrics=['mse', cls.correlation_coefficient_loss])
        return model

    def get_model(self, method = setting.model_type):

        crispr_model = getattr(self, "_{!s}__{!s}_model".format(self.__class__.__name__, method), self.__mlp_model)(
            setting.FC_layout, self.input_len)
        return self.compile_transfer_learning_model(crispr_model)

