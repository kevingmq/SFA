package sfa.transformation;

import com.carrotsearch.hppc.IntIntHashMap;
import sfa.classification.Classifier.Words;
import sfa.classification.ParallelFor;

import java.util.concurrent.atomic.AtomicInteger;

public class BOSSMD {

    public int alphabetSize;
    public int maxF;

    public int windowLength;
    public boolean normMean;
    //public boolean lowerBounding;
    public SFA[] signature;

    public final static int BLOCKS;

    static {
        Runtime runtime = Runtime.getRuntime();
        if (runtime.availableProcessors() <= 4) {
            BLOCKS = 8;
        } else {
            BLOCKS = runtime.availableProcessors();
        }

        //    BLOCKS = 1; // for testing purposes
    }

    public BOSSMD() {
    }

    /**
     * Create a BOSS MD.
     *
     * @param maxF          queryLength of the SFA words
     * @param maxS          alphabet size
     * @param windowLength  sub-sequence (window) queryLength used for extracting SFA words from
     *                      time series.
     * @param normMean      set to true, if mean should be set to 0 for a window
     * @param lowerBounding set to true, if the Fourier transform should be normed (typically
     *                      used to lower bound / mimic Euclidean distance).
     */
    public BOSSMD(int maxF, int maxS, int windowLength, boolean normMean) {
        this.maxF = maxF;
        this.symbols = maxS;
        this.windowLength = windowLength;
        this.normMean = normMean;
    }

    /**
     * The BOSS MD: a histogram of SFA word frequencies
     */
    public static class BagOfPatternMD {
        public IntIntHashMap bag;
        public Double label;

        public BagOfPatternMD() {
        }

        public BagOfPatternMD(int size, Double label) {
            this.bag = new IntIntHashMap(size);
            this.label = label;
        }
    }


    /**
     * Create SFA words for all samples
     *
     * @param samples the time series to be transformed
     * @return returns an array of words for each time series
     */
    public int[][] createMDWords(final TimeSeries[] samples) {

        final int[][] words = new int[samples.length][];

        if (this.signature == null) {
            this.signature = new SFA(HistogramType.EQUI_DEPTH);
            this.signature.fitWindowing(samples, this.windowLength, this.maxF, this.symbols, this.normMean, true);
        }

        // create sliding windows
        ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
            @Override
            public void run(int id, AtomicInteger processed) {
                for (int i = 0; i < samples.length; i++) {
                    if (i % BLOCKS == id) {
                        short[][] sfaWords = BOSS.this.signature.transformWindowing(samples[i]);
                        words[i] = new int[sfaWords.length];
                        for (int j = 0; j < sfaWords.length; j++) {
                            words[i][j] = (int) Words.createWord(sfaWords[j], BOSS.this.maxF, (byte) Words.binlog(BOSS.this.symbols));
                        }
                    }
                }
            }
        });

        return words;
    }

    /**
     * Create the BOSS boss for a fixed window-queryLength and SFA word queryLength
     *
     * @param words      the SFA words of the time series
     * @param samples    the samples to be transformed
     * @param wordLength the SFA word queryLength
     * @return returns a BOSS boss for each time series in samples
     */
    public BagOfPattern[] createBagOfPattern(
            final int[][] words,
            final TimeSeries[] samples,
            final int wordLength) {
        BagOfPattern[] bagOfPatterns = new BagOfPattern[words.length];

        final byte usedBits = (byte) Words.binlog(this.symbols);
        // FIXME
        // final long mask = (usedBits << wordLength) - 1l;
        final long mask = (1L << (usedBits * wordLength)) - 1L;

        // iterate all samples
        for (int j = 0; j < words.length; j++) {
            bagOfPatterns[j] = new BagOfPattern(words[j].length, samples[j].getLabel());

            // create subsequences
            long lastWord = Long.MIN_VALUE;

            for (int offset = 0; offset < words[j].length; offset++) {
                // use the words of larger queryLength to get words of smaller lengths
                long word = words[j][offset] & mask;
                if (word != lastWord) { // ignore adjacent samples
                    bagOfPatterns[j].bag.putOrAdd((int) word, (short) 1, (short) 1);
                }
                lastWord = word;
            }
        }

        return bagOfPatterns;
    }
}
