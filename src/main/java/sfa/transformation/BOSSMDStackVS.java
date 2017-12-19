package sfa.transformation;

import com.carrotsearch.hppc.*;
import com.carrotsearch.hppc.cursors.*;
import sfa.classification.Classifier;
import sfa.classification.ParallelFor;
import sfa.timeseries.MultiVariateTimeSeries;
import sfa.timeseries.TimeSeries;
import sfa.transformation.MUSE.Dictionary;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
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
        public IntIntHashMap bag;
        public Double label;

        public BagOfPattern(int size, Double label) {
            this.bag = new IntIntHashMap(size);
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

            BagOfPattern bop = new BagOfPattern(100, samples[index].getLabel());
            for (int idSource = 0; idSource < dimensionality; idSource++) {

                // create subsequences
                String dLabel = String.valueOf(idSource);

                for (int offset = 0; offset < words[index][idSource].length; offset++) {
                    String word = dLabel + "_" +((words[index][idSource][offset] & mask));
                    int dict = this.dict.getWord(word);
                    bop.bag.putOrAdd(dict,1,1);
                }
            }
            bagOfPatterns.add(bop);
        }

        return bagOfPatterns.toArray(new BagOfPattern[]{});
    }

    public ObjectObjectHashMap<Double, IntFloatHashMap> createTfIdf(
            final BagOfPattern[] bagOfPatterns,
            final Set<Double> uniqueLabels) {
        int[] sampleIndices = createIndices(bagOfPatterns.length);
        return createTfIdf(bagOfPatterns, sampleIndices, uniqueLabels);
    }

    protected static int[] createIndices(int length) {
        int[] indices = new int[length];
        for (int i = 0; i < length; i++) {
            indices[i] = i;
        }
        return indices;
    }

    public ObjectObjectHashMap<Double, IntFloatHashMap> createTfIdf(
            final BagOfPattern[] bagOfPatterns,
            final int[] sampleIndices,
            final Set<Double> uniqueLabels) {

        ObjectObjectHashMap<Double, IntFloatHashMap> matrix = new ObjectObjectHashMap<>(
                uniqueLabels.size());
        initMatrix(matrix, uniqueLabels, bagOfPatterns);

        for (int j : sampleIndices) {
            Double label = bagOfPatterns[j].label;
            IntFloatHashMap wordInBagFreq = matrix.get(label);
            for (IntIntCursor key : bagOfPatterns[j].bag) {
                wordInBagFreq.putOrAdd(key.key, key.value, key.value);
            }
        }

        // count the number of classes where the word is present
        IntIntHashMap wordInClassFreq = new IntIntHashMap(matrix.iterator().next().value.size());

        for (ObjectCursor<IntFloatHashMap> stat : matrix.values()) {
            // count the occurrence of words
            for (IntFloatCursor key : stat.value) {
                wordInClassFreq.putOrAdd(key.key, (short) 1, (short) 1);
            }
        }

        // calculate the tfIDF value for each class
        for (ObjectObjectCursor<Double, IntFloatHashMap> stat : matrix) {
            IntFloatHashMap tfIDFs = stat.value;
            // calculate the tfIDF value for each word
            for (IntFloatCursor patternFrequency : tfIDFs) {
                int wordCount = wordInClassFreq.get(patternFrequency.key);
                if (patternFrequency.value > 0
                        && uniqueLabels.size() != wordCount // avoid Math.log(1)
                        ) {
                    double tfValue = 1.0 + Math.log10(patternFrequency.value); // smoothing
                    double idfValue = Math.log10(1.0 + uniqueLabels.size() / (double) wordCount); // smoothing
                    double tfIdf = tfValue / idfValue;

                    // update the tfIDF vector
                    tfIDFs.values[patternFrequency.index] = (float) tfIdf;
                } else {
                    tfIDFs.values[patternFrequency.index] = 0;
                }
            }
        }

        // norm the tf-idf-matrix
        normalizeTfIdf(matrix);

        return matrix;
    }

    protected void initMatrix(
            final ObjectObjectHashMap<Double, IntFloatHashMap> matrix,
            final Set<Double> uniqueLabels,
            final BagOfPattern[] bag) {
        int maxElements = (int) (dict.size() * 0.75 + 1);
        for (Double label : uniqueLabels) {
            IntFloatHashMap stat = matrix.get(label);
            if (stat == null) {
                matrix.put(label, new IntFloatHashMap(maxElements));
            } else {
                stat.clear();
            }
        }
    }

    public void normalizeTfIdf(final ObjectObjectHashMap<Double, IntFloatHashMap> classStatistics) {
        for (ObjectCursor<IntFloatHashMap> classStat : classStatistics.values()) {
            double squareSum = 0.0;
            for (FloatCursor entry : classStat.value.values()) {
                squareSum += entry.value * entry.value;
            }
            double squareRoot = Math.sqrt(squareSum);
            if (squareRoot > 0) {
                for (FloatCursor entry : classStat.value.values()) {
                    //entry.value /= squareRoot;
                    classStat.value.values[entry.index] /= squareRoot;
                }
            }
        }
    }
}
