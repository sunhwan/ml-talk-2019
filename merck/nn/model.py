import luigi
import numpy as np

def Rsquared(x, y):
    avx = np.average(x)
    avy = np.average(y)
    num = np.sum( (x-avx)*(y-avy) )**2
    denom = np.sum( (x-avx)**2 ) * np.sum( (y-avy)**2 )
    return num/denom

class Model(luigi.Task):
    dataset = luigi.Parameter()

    def input(self):
        return {
            'train': luigi.LocalTarget('../merck.data/csv/%s_training_disguised.csv' % self.dataset),
            'test': luigi.LocalTarget('../merck.data/csv/%s_test_disguised.csv' % self.dataset)
            }

    def output(self):
        return luigi.LocalTarget('csv/%s_prediction.csv' % self.dataset)

    def run(self):
        import pandas as pd
        import numpy as np
        from keras.models import Sequential
        from keras.optimizers import SGD
        from keras.layers import Dense, Activation, Dropout
        from keras.regularizers import l2

        train = pd.read_csv(self.input()['train'].path)
        test = pd.read_csv(self.input()['test'].path)

        # only use classifiers exists in both train and test sets
        cols = set(train.columns)
        cols = cols.intersection(set(test.columns))

        # use first 75% of data; time-split
        #l = len(train)
        #train_x = train[:int(l*0.75)].filter(cols).drop(['MOLECULE', 'Act'], axis=1)
        #train_y = train[:int(l*0.75)].filter(['Act']).values.ravel()

        # .. but because we are not doing cross-validation, we will use all available data
        train_x = train.filter(cols).drop(['MOLECULE', 'Act'], axis=1).values
        train_y = train.filter(['Act']).values
        test_x = test.filter(cols).drop(['MOLECULE', 'Act'], axis=1).values
        test_y = test.filter(['Act']).values

        # transformation
        train_x = np.log10(train_x + 1)
        test_x = np.log10(test_x + 1)

        # DNN model with the recommended parameters
        input_dim = train_x.shape[1]
        model = Sequential([
            Dense(4000, input_dim=input_dim, kernel_initializer='lecun_uniform'),
            Activation('relu'),
            Dropout(0.25),
        
            Dense(2000, kernel_initializer='lecun_uniform'),
            Activation('relu'),
            Dropout(0.25),
        
            Dense(1000, kernel_initializer='lecun_uniform'),
            Activation('relu'),
            Dropout(0.25),
        
            Dense(1000, kernel_initializer='lecun_uniform'),
            Activation('relu'),
            Dropout(0.1),
        
            Dense(1, kernel_initializer='lecun_uniform')
        ])

        sgd = SGD()
        model.compile(loss='mean_squared_error', optimizer=sgd)
        model.fit(train_x, train_y, epochs=50, batch_size=32)
        y_pred = model.predict(test_x).ravel()

        # R squared value
        print(self.dataset, Rsquared(test_y, y_pred))

        # write to output
        prediction = pd.DataFrame(data={"Act": y_pred})
        prediction.to_csv(self.output().path)


class FitAllDataset(luigi.WrapperTask):
    def requires(self):
        datasets = ('3A4', 'CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'LOGD', 'METAB', 'NK1', 'OX1', 'OX2', 'PGP', 'PPB', 'RAT_F', 'TDI', 'THROMBIN')
        for dataset in datasets:
            yield Model(dataset=dataset)


if __name__ == '__main__':
    import logging
    #luigi.interface.setup_interface_logging()
    #logger = luigi.interface.logging.getLogger('luigi-interface')
    #logger.setLevel(logging.ERROR)

    #luigi.run('--dataset CB1'.split(), main_task_cls=Model, local_scheduler=True)
    luigi.run([], main_task_cls=FitAllDataset, local_scheduler=True)
