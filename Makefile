.PHONY: data
data:
	mkdir -p data
	curl -L -o data/LJSpeech-1.1.tar.bz2 https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
	tar -xjf data/LJSpeech-1.1.tar.bz2 -C data
	rm data/LJSpeech-1.1.tar.bz2
