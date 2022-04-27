from evaluation_scripts.cfid.embeddings import InceptionEmbedding
from evaluation_scripts.cfid.cfid_metric import CFIDMetric
from data_loaders.prepare_data import create_test_loader
from wrappers.our_gen_wrapper import load_best_gan


def get_cfid(args):
    print("GETTING INCEPTION EMBEDDING")
    inception_embedding = InceptionEmbedding(parallel=True)
    print("GETTING GENERATOR")
    G = load_best_gan(args)
    G.update_gen_status(val=True)

    print("GETTING DATA LOADERS")
    loader = create_test_loader(args)
    cfid_metric = CFIDMetric(gan=G,
                             loader=loader,
                             image_embedding=inception_embedding,
                             condition_embedding=inception_embedding,
                             cuda=True,
                             args=args)

    print('CFID: ', cfid_metric.get_cfid_torch())
