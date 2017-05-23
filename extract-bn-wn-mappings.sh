cd BabelNet-API-3.7

CP=babelnet-api-3.7.jar:config:`echo lib/*.jar | tr ' ' ':'`

javac -cp $CP ExtractMappings.java && \
		java -cp .:$CP ExtractMappings | tee ../output/bn-wn-mappings.txt