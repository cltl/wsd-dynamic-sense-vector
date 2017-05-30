cd gensim-modified
cython gensim/models/word2vec_inner.pyx gensim/models/doc2vec_inner.pyx && \
	python3 setup.py build_ext --inplace && \
	python3 setup.py install --user
	#python3 setup.py test 