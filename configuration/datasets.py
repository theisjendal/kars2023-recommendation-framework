from shared.configuration_classes import DatasetConfiguration, CountFilter
from shared.enums import Sentiment

# All users have at least 5 positive ratings, for 5 folds.
mindreader = DatasetConfiguration('mindreader', lambda x: {x > 0: 1}.get(True, 0),
                                  filters=[CountFilter(lambda c: c >= 5, sentiment=Sentiment.POSITIVE)])
movielens = DatasetConfiguration('movielens', lambda x: {x < 3: -1, x > 3: 1}.get(True, 0),
                                 filters=[CountFilter(lambda c: c >= 5, sentiment=Sentiment.POSITIVE)])
ml_mr = DatasetConfiguration('ml-mr', lambda x: {x > 3: 1}.get(True, 0),
                             filters=[CountFilter(lambda c: c >= 5, sentiment=Sentiment.POSITIVE)])
ml_mr_1m = DatasetConfiguration('ml-mr-1m', lambda x: {x > 3: 1}.get(True, 0),
                                filters=[CountFilter(lambda c: c >= 5, sentiment=Sentiment.POSITIVE)],
                                max_users=12500)
synthetic = DatasetConfiguration('synthetic', lambda x: {x >= 1: 1}.get(True, 0),
                                 filters=[CountFilter(lambda c: c >= 5, sentiment=Sentiment.POSITIVE)])
synthetic_1m = DatasetConfiguration('synthetic-1m', lambda x: {x >= 1: 1}.get(True, 0),
                                    filters=[CountFilter(lambda c: c >= 5, sentiment=Sentiment.POSITIVE)],
                                    max_users=12500)
synthetic_16 = DatasetConfiguration('synthetic-16', lambda x: {x >= 1: 1}.get(True, 0),
                                    filters=[CountFilter(lambda c: c >= 5, sentiment=Sentiment.POSITIVE)])
dbbook = DatasetConfiguration('dbbook', lambda x: {x >= 3: 1}.get(True, 0),
                              filters=[CountFilter(lambda c: c >= 5, sentiment=Sentiment.POSITIVE)])
lastfm = DatasetConfiguration('lastfm', lambda x: x,
                              filters=[CountFilter(lambda c: c >= 5, sentiment=Sentiment.POSITIVE)])
amazon_book = DatasetConfiguration('amazon-book', lambda x: x, k_core=1)
amazon_book_s = DatasetConfiguration('amazon-book-s', lambda x: x, max_users=60000, k_core=1)
amazon_book_simplex = DatasetConfiguration('amazon-book-simplex', lambda x: x,
                              filters=[CountFilter(lambda c: c >= 5, sentiment=Sentiment.POSITIVE)])
amazon_beauty = DatasetConfiguration('amazon-beauty-idcf', lambda x: x,
                                     filters=[CountFilter(lambda c: c >= 5, sentiment=Sentiment.POSITIVE)])

datasets = [mindreader, movielens, ml_mr, ml_mr_1m, synthetic, synthetic_1m, synthetic_16, lastfm, amazon_book,
            amazon_book_simplex, amazon_beauty, amazon_book_s]
dataset_names = [d.name for d in datasets]