import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam
from DataHandler import LoadData, negSamp, transpose, transToLsts
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Recommender:
	def __init__(self, sess, datas):
		self.sess = sess
		self.trnMats, self.tstInt, self.label, self.tstUsrs, args.intTypes = datas

		self.tpMats = list(map(transpose, self.trnMats))
		args.user, args.item = self.trnMats[0].shape
		print('USER', args.user, 'ITEM', args.item)
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'HR', 'NDCG']
		for met in mets:
			self.metrics['Train'+met] = list()
			self.metrics['Test'+met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for met in reses:
			val = reses[met]
			ret += '%s = %.4f, ' % (met, val)
			tem = name + met
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		if args.load_model != None:
			self.loadModel()
			stloc = len(self.metrics['TrainLoss'])
		else:
			stloc = 0
			init = tf.global_variables_initializer()
			self.sess.run(init)
			log('Variables Inited')
		for ep in range(stloc, args.epoch):
			test = (ep % args.test_epoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, test))
			if test:
				reses = self.testEpoch()
				log(self.makePrint('Test', ep, reses, test))
			if ep % 5 == 0:
				self.saveHistory()
			print()
		reses = self.testEpoch()
		log(self.makePrint('Test', args.epoch, reses, True))
		self.saveHistory()

	def encoder(self, embed, vecs):
		unAct = []
		catlat1 = []
		paramId = 'dfltP%d' % NNs.getParamId()
		for vec in vecs:
			temlat = tf.sparse.sparse_dense_matmul(vec, embed)
			memoatt = FC(temlat, args.memosize, activation='relu', reg=True, useBias=True)
			memoTrans = tf.reshape(FC(memoatt, args.latdim**2, reg=True, name=paramId, reuse=True), [-1, args.latdim, args.latdim])
			transLat = tf.reduce_sum(tf.reshape(temlat, [-1, args.latdim, 1]) * memoTrans, axis=1)
			unAct.append(transLat)
			catlat1.append(tf.nn.relu(transLat))
		catlat2 = NNs.selfAttention(catlat1, number=args.intTypes, inpDim=args.latdim, numHeads=args.attHead)
		# aggregation
		globalQuery = tf.reduce_sum(tf.stack(unAct, axis=-1), axis=-1)
		weights = []
		paramId = 'dfltP%d' % NNs.getParamId()
		for catlat in catlat2:
			temlat = FC(tf.concat([globalQuery, catlat], axis=-1), args.latdim//2, useBias=True, reg=True, activation='relu', name=paramId+'_1', reuse=True)
			weight = FC(temlat, 1, useBias=True, reg=True, name=paramId+'_2', reuse=True)
			weights.append(weight)
		stkWeight = tf.concat(weights, axis=1)
		sftWeight = tf.reshape(tf.nn.softmax(stkWeight*32, axis=1), [-1, args.intTypes, 1])
		stkCatlat = tf.stack(catlat2, axis=1)
		lat = tf.reshape(tf.reduce_sum(sftWeight * stkCatlat, axis=1), [-1, args.latdim])
		for i in range(args.deep_layer):
			lat = FC(lat, args.latdim, reg=True, useBias=True, activation='relu') + lat
		return lat

	def predict(self, alats, aids, bids, bname):
		pckLats = tf.nn.embedding_lookup(alats, aids)
		num = args.user if bname=='user' else args.item
		blats = NNs.defineParam(bname+'_embed', shape=[num, args.latdim], dtype=tf.float32, reg=True)
		pckALats = tf.nn.embedding_lookup(alats, aids)
		pckBLats = tf.nn.embedding_lookup(blats, bids)
		return tf.reduce_sum(pckALats * pckBLats, axis=-1) * args.mult

	def ours(self):
		UEmbed = NNs.defineParam('UEmbed', shape=[args.user, args.latdim], dtype=tf.float32, reg=True)
		IEmbed = NNs.defineParam('IEmbed', shape=[args.item, args.latdim], dtype=tf.float32, reg=True)
		ulats = self.encoder(IEmbed, self.uVecs)
		ilats = self.encoder(UEmbed, self.iVecs)

		uPred = self.predict(ulats, self.uids, self.iids, 'item')
		iPred = self.predict(ilats, self.iids, self.uids, 'user')
		pred = uPred + iPred
		return pred

	def prepareModel(self):
		self.uVecs = []
		self.iVecs = []
		for i in range(args.intTypes):
			idx, data, shape = transToLsts(self.trnMats[i])
			self.uVecs.append(tf.sparse.SparseTensor(idx, data, shape))
			idx, data, shape = transToLsts(self.tpMats[i])
			self.iVecs.append(tf.sparse.SparseTensor(idx, data, shape))
		self.uids = tf.placeholder(name='uids', dtype=tf.int32, shape=[None])
		self.iids = tf.placeholder(name='iids', dtype=tf.int32, shape=[None])

		self.pred = self.ours()
		sampNum = tf.shape(self.iids)[0] // 2
		posPred = tf.slice(self.pred, [0], [sampNum])
		negPred = tf.slice(self.pred, [sampNum], [-1])
		self.preLoss = tf.reduce_sum(tf.maximum(0.0, 1.0 - (posPred - negPred))) / args.batch
		self.regLoss = args.reg * Regularize()
		self.loss = self.preLoss + self.regLoss

		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

	def sampleTrainBatch(self, batIds):
		# pre-sample for each epoch, in order to accelarate
		preSamp = list(np.random.permutation(args.item))
		temLabel = self.label[batIds].toarray()
		batch = len(batIds)
		temlen = batch * 2 * args.sampNum
		uLocs = [None] * temlen
		iLocs = [None] * temlen
		cur = 0
		for i in range(batch):
			posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
			negset = negSamp(temLabel[i], preSamp)
			poslocs = np.random.choice(posset, args.sampNum)
			neglocs = np.random.choice(negset, args.sampNum)
			for j in range(args.sampNum):
				posloc = poslocs[j]
				negloc = neglocs[j]
				uLocs[cur] = uLocs[cur+temlen//2] = batIds[i]
				iLocs[cur] = posloc
				iLocs[cur+temlen//2] = negloc
				cur += 1
		return uLocs, iLocs

	def trainEpoch(self):
		num = args.user
		sfIds = np.random.permutation(num)[:args.trnNum]
		epochLoss, epochPreLoss = [0] * 2
		num = len(sfIds)
		steps = int(np.ceil(num / args.batch))

		for i in range(steps):
			st = i * args.batch
			ed = min((i+1) * args.batch, num)
			batIds = sfIds[st: ed]

			uLocs, iLocs = self.sampleTrainBatch(batIds)
			target = [self.optimizer, self.preLoss, self.regLoss, self.loss]
			res = self.sess.run(target, feed_dict={self.uids: uLocs, self.iids: iLocs}, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

			preLoss, regLoss, loss = res[1:]

			epochLoss += loss
			epochPreLoss += preLoss
			log('Step %d/%d: loss = %.2f, regLoss = %.2f          ' % (i, steps, loss, regLoss), save=False, oneline=True)
		ret = dict()
		ret['Loss'] = epochLoss / steps
		ret['preLoss'] = epochPreLoss / steps
		return ret

	def sampleTestBatch(self, batIds):
		batch = len(batIds)
		temTst = self.tstInt[batIds]
		temLabel = self.label[batIds].toarray()
		temlen = batch * 100
		uLocs = [None] * temlen
		iLocs = [None] * temlen
		tstLocs = [None] * batch
		cur = 0
		for i in range(batch):
			posloc = temTst[i]
			negset = np.reshape(np.argwhere(temLabel[i]==0), [-1])
			rdnNegSet = np.random.permutation(negset)[:99]
			locset = np.concatenate((rdnNegSet, np.array([posloc])))
			tstLocs[i] = locset
			for j in range(100):
				uLocs[cur] = batIds[i]
				iLocs[cur] = locset[j]
				cur += 1
		return uLocs, iLocs, temTst, tstLocs

	def testEpoch(self):
		epochHit, epochNdcg = [0] * 2
		ids = self.tstUsrs
		num = len(ids)
		print(len(ids))
		testbatch = np.maximum(1, args.batch * args.sampNum * 2 // 100)
		steps = int(np.ceil(num / testbatch))
		for i in range(steps):
			st = i * testbatch
			ed = min((i+1) * testbatch, num)
			batIds = ids[st: ed]
			uLocs, iLocs, temTst, tstLocs = self.sampleTestBatch(batIds)

			preds = self.sess.run(self.pred, feed_dict={self.uids: uLocs, self.iids: iLocs}, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
			hit, ndcg = self.calcRes(np.reshape(preds, [ed-st, 100]), temTst, tstLocs)
			epochHit += hit
			epochNdcg += ndcg
			log('Step %d/%d: hit = %d, ndcg = %d               ' % (i, steps, hit, ndcg), save=False, oneline=True)
		ret = dict()
		ret['HR'] = epochHit / num
		ret['NDCG'] = epochNdcg / num
		return ret

	def calcRes(self, preds, temTst, tstLocs):
		hit, ndcg = [0] * 2
		for i in range(preds.shape[0]):
			predvals = list(zip(preds[i], tstLocs[i]))
			predvals.sort(key=lambda x: x[0], reverse=True)
			shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
			if temTst[i] in shoot:
				hit += 1
				ndcg += np.reciprocal(np.log2(shoot.index(temTst[i]) + 2))
		return hit, ndcg

	def saveHistory(self):
		if args.epoch == 0:
			return
		with open('History/' + args.save_path + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)

		saver = tf.train.Saver()
		saver.save(self.sess, 'Models/' + args.save_path)
		log('Model Saved: %s' % args.save_path)

	def loadModel(self):
		saver = tf.train.Saver()
		saver.restore(sess, 'Models/' + args.load_model)
		with open('History/' + args.load_model + '.his', 'rb') as fs:
		    self.metrics = pickle.load(fs)
		log('Model Loaded')

if __name__ == '__main__':
	logger.saveDefault = True
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	log('Start')
	datas = LoadData()
	log('Load Data')

	with tf.Session(config=config) as sess:
		recom = Recommender(sess, datas)
		recom.run()
