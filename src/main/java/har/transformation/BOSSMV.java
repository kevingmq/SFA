package har.transformation;

import com.carrotsearch.hppc.LongIntHashMap;
import sfa.classification.Classifier.Words;
import sfa.classification.ParallelFor;
import har.timeseries.MultiDimTimeSeries;
import sfa.timeseries.TimeSeries;
import sfa.transformation.BOSSVS;
import sfa.transformation.SFA;

import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicInteger;

public class BOSSMV extends BOSSVS {

    public SFA[] signatures;

    public static int BLOCKS;

    static {
        Runtime runtime = Runtime.getRuntime();
        if (runtime.availableProcessors() <= 4) {
            BLOCKS = 8;
        } else {
            BLOCKS = runtime.availableProcessors();
        }

        //BLOCKS = 1; // for testing purposes
    }

    public BOSSMV() {
    }

    /**
     * Create a BOSS MD.
     *
     * @param maxF         queryLength of the SFA words
     * @param maxS         alphabet size
     * @param windowLength sub-sequence (window) queryLength used for extracting SFA words from
     *                     time series.
     * @param normMean     set to true, if mean should be set to 0 for a window
     */
    public BOSSMV(int maxF, int maxS, int windowLength, boolean normMean) {
        this.maxF = maxF;
        this.symbols = maxS;
        this.windowLength = windowLength;
        this.normMean = normMean;
    }

    /**
     * The BOSS MD: a histogram of MD word frequencies
     */
    public static class BagOfPatternMD {
        public LongIntHashMap bag;
        public Double label;

        public BagOfPatternMD(int size, Double label) {
            this.bag = new LongIntHashMap(size);
            this.label = label;
        }
    }

    private TimeSeries[][] splitMultiDimTimeSeries(int numSources, MultiDimTimeSeries[] samples) {

        TimeSeries[][] samplesModificado = new TimeSeries[numSources][samples.length];
        for (int indexOfSource = 0; indexOfSource < numSources; indexOfSource++) {

            for (int indexOfSample = 0; indexOfSample < samples.length; indexOfSample++) {
                samplesModificado[indexOfSource][indexOfSample] = samples[indexOfSample].getTimeSeriesOfOneSource(indexOfSource);
            }
        }
        return samplesModificado;
    }

    /**
     * Create MD words for all samples
     *
     * @param samples the time series to be transformed
     * @return returns an array of words for each time series
     */
    public int[][] createMDWords(final MultiDimTimeSeries[] samples) {

        final int numSources = samples[0].getNumSources();
        final int[][] mdWordsInt = new int[samples.length][];
        final short[][][][] words = new short[samples.length][numSources][][];


        TimeSeries[][] samplesSplited = splitMultiDimTimeSeries(numSources, samples);

        if (this.signatures == null) {
            this.signatures = new SFA[numSources];
            for (int idSource = 0; idSource < numSources; idSource++) {
                this.signatures[idSource] = new SFA(SFA.HistogramType.EQUI_DEPTH);
                this.signatures[idSource].fitWindowing(samplesSplited[idSource], this.windowLength, this.maxF, this.symbols, this.normMean, true);
            }
        }

        // create sliding windows
        ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
            @Override
            public void run(int id, AtomicInteger processed) {
                for (int i = 0; i < samples.length; i++) {
                    if (i % BLOCKS == id) {

                        for (int idSource = 0; idSource < numSources; idSource++) {
                            short[][] sfaWords = BOSSMV.this.signatures[idSource].transformWindowing(samplesSplited[idSource][i]);
                            words[i][idSource] = sfaWords;

                        }

                        short[][] mdWordsmerged = mergeSFAWords(words[i], numSources);

                        mdWordsInt[i] = new int[mdWordsmerged.length];
                        for (int j = 0; j < mdWordsmerged.length; j++) {
                            mdWordsInt[i][j] = (int) Words.createWord(mdWordsmerged[j], BOSSMV.this.maxF, (byte) Words.binlog(BOSSMV.this.symbols));
                        }
                    }
                }
            }
        });

        return mdWordsInt;
    }

    short[][] mergeSFAWords(short[][][] words, int numSources) {
        int countOfWords = words[0].length;
        int wordLength = words[0][0].length;

        short[][] result;
        ArrayList<short[]> mylist = new ArrayList<>();
        for (int i = 0; i < countOfWords; i++) {
            short[][] output = merded(words, numSources, i,wordLength);
            for(int j =0; j < output.length; j++){
                mylist.add(output[j]);
            }
        }
        result = mylist.toArray(new short[][]{});
       /* for (int idSource = 0; idSource < numSources; idSource++) {

                short[][] t = words[idSource];

                mdWords[idSource] = new int[twords.length];
                for (int k = 0; k < twords.length; k++) {
                    mdWords[idSource][k] = (int) Words.createWord(twords[k], BOSSMV.this.maxF, (byte) Words.binlog(BOSSMV.this.alphabetSize));
                }

        }*/

        return result;
    }

    short[][] merded(short[][][] words, int numSources, int i_index,int lenghtWord) {

        int lenght = numSources*lenghtWord;
        short[][] mdword = new short[1][lenght];
        int count = 0;

        for(int j = 0; j < lenghtWord; j++){
            for (int idSource = 0; idSource < numSources; idSource++) {
                short c = words[idSource][i_index][j];
                mdword[0][count] = c;
                count++;
            }
        }

        return mdword;
    }

    /**
     * Create the BOSS boss for a fixed window-queryLength and SFA word queryLength
     *
     * @param mdWords    the SFA words of the time series
     * @param samples    the samples to be transformed
     * @param wordLength the SFA word queryLength
     * @return returns a BOSS boss for each time series in samples
     */
    public BagOfPatternMD[] createBagOfPattern(
            final int[][] mdWords,
            final MultiDimTimeSeries[] samples,
            final int wordLength) {
        BagOfPatternMD[] bagOfPatterns = new BagOfPatternMD[mdWords.length];

        final byte usedBits = (byte) Words.binlog(this.symbols);
        // FIXME
        // final long mask = (usedBits << wordLength) - 1l;
        final long mask = (1L << (usedBits * wordLength)) - 1L;

        // iterate all samples
        for (int j = 0; j < mdWords.length; j++) {
            bagOfPatterns[j] = new BagOfPatternMD(mdWords[j].length, samples[j].getLabel());

            // create subsequences
            long lastWord = Long.MIN_VALUE;

            for (int offset = 0; offset < mdWords[j].length; offset++) {
                // use the words of larger queryLength to get words of smaller lengths
                long word = mdWords[j][offset] & mask;
                if (word != lastWord) { // ignore adjacent samples
                    bagOfPatterns[j].bag.putOrAdd((int) word, (short) 1, (short) 1);
                }
                //lastWord = word;
            }
        }

        return bagOfPatterns;
    }
}
