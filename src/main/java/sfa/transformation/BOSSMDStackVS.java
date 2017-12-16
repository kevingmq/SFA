package sfa.transformation;

import com.carrotsearch.hppc.IntLongHashMap;
import com.carrotsearch.hppc.LongIntHashMap;
import sfa.classification.Classifier;
import sfa.classification.ParallelFor;
import sfa.timeseries.MultiVariateTimeSeries;
import sfa.timeseries.TimeSeries;
import sfa.transformation.MUSE.Dictionary;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class BOSSMDStackVS {

    public int alphabetSize;
    public int maxF;
    public SFA.HistogramType histogramType = null;

    public int windowLength;
    public boolean normMean;
    public SFA[] signatures;
    public Dictionary dict;

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

    public BOSSMDStackVS(int maxF, int maxS, int windowLength, boolean normMean) {
        this.maxF = maxF + maxF % 2;
        this.alphabetSize = maxS;
        this.windowLength = windowLength;
        this.normMean = normMean;
        this.dict = new Dictionary();
    }

    public static class BagOfPattern {
        public IntLongHashMap bag;
        public Double label;

        public BagOfPattern(int size, Double label) {
            this.bag = new IntLongHashMap(size);
            this.label = label;
        }
    }

    public static TimeSeries[][] splitMultiDimTimeSeries(int numSources,final MultiVariateTimeSeries[] samples) {

        TimeSeries[][] samplesModificado = new TimeSeries[numSources][samples.length];
        for (int indexOfSource = 0; indexOfSource < numSources; indexOfSource++) {

            for (int indexOfSample = 0; indexOfSample < samples.length; indexOfSample++) {
                samplesModificado[indexOfSource][indexOfSample] = samples[indexOfSample].getTimeSeriesOfOneSource(indexOfSource);
            }
        }
        return samplesModificado;
    }

    public int[/*samples */][/*source */][/*wordInt*/] createWords(final MultiVariateTimeSeries[] samples) {
        final int numSources = samples[0].getDimensions();
        TimeSeries[][] samplesSplited = splitMultiDimTimeSeries(numSources, samples);

        // SFA quantization
        if (this.signatures == null) {
            this.signatures = new SFA[numSources];
            for (int idSource = 0; idSource < numSources; idSource++) {
                this.signatures[idSource] = new SFA(SFA.HistogramType.EQUI_DEPTH);
                this.signatures[idSource].fitWindowing(samplesSplited[idSource], this.windowLength, this.maxF, this.alphabetSize, this.normMean, true);
            }

        }
        // create bag of words for each sample
        final int[][][] words = new int[samples.length][numSources][];
        ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
            @Override
            public void run(int id, AtomicInteger processed) {
                for (int i = 0; i < samples.length; i++) {
                    if (i % BLOCKS == id) {
                        words[i] = createWords(samples[i]);
                    }
                }
            }
        });
        return words;
    }

    private int[/*source*/][/*wordInt*/] createWords(final MultiVariateTimeSeries sample) {

        final int numSources = sample.getDimensions();
        final int[][] words = new int[numSources][];

        // create sliding windows -> words

        for (int idSource = 0; idSource < numSources; idSource++) {
            words[idSource] = signatures[idSource].transformWindowingInt(sample.getTimeSeriesOfOneSource(idSource), this.maxF);

        }

        return words;
    }

    public BagOfPattern[] createBagOfPattern(
            final int[/*samples */][/*source */][/*wordInt*/] words,
            final MultiVariateTimeSeries[] samples,
            final int wordLength) {
        List<BagOfPattern> bagOfPatterns = new ArrayList<BagOfPattern>(samples.length);

        final int dimensionality = samples[0].getDimensions();

        final byte usedBits = (byte) Classifier.Words.binlog(this.alphabetSize);

        //    final long mask = (usedBits << wordLength) - 1l;
        final long mask = (1L << (usedBits * wordLength)) - 1L;

        // iterate all samples
        for(int index = 0; index < samples.length; index++){
            for (int idSource = 0; idSource < dimensionality; idSource++) {
                BagOfPattern bop = new BagOfPattern(100, samples[index].getLabel());

                // create subsequences

                String dLabel = String.valueOf(idSource);

                for (int offset = 0; offset < words[index][idSource].length; offset++) {
                    String word = dLabel + "_" +((words[index][idSource][offset] & mask));
                    int dict = this.dict.getWord(word);
                    bop.bag.putOrAdd(dict,1,1);
                }
                bagOfPatterns.add(bop);
            }
        }

        return bagOfPatterns.toArray(new BagOfPattern[]{});
    }


}
