package sfa.main;

import com.carrotsearch.hppc.IntFloatHashMap;
import com.carrotsearch.hppc.ObjectIntHashMap;
import com.carrotsearch.hppc.ObjectObjectHashMap;
import com.carrotsearch.hppc.cursors.IntIntCursor;
import com.carrotsearch.hppc.cursors.ObjectObjectCursor;
import sfa.classification.Classifier;
import sfa.timeseries.MultiVariateTimeSeries;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;
import sfa.transformation.BOSSMDStackVS;
import sfa.transformation.SFA;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;


import static sfa.transformation.BOSSMDStackVS.splitMultiDimTimeSeries;

public class TestPerformace {
    public static void main (String[] args) {

        //Fit SFA
        boolean DEBUG = true;
        Classifier.DEBUG = DEBUG;
        TimeSeriesLoader.DEBUG = DEBUG;

        String home = System.getProperty("user.home") + "/datasets";

        File dir = new File(home);

        File train = new File(dir.getAbsolutePath() + "/" + "dataset2" );

        File test = new File(dir.getAbsolutePath() + "/" + "dataset1" );


        int num_sources = 3;
        int segment_length = 151;
        int windowLenght = 30;
        int wordLenght = 16;
        int symbols = 4;
        boolean normMean = true;
        boolean lowerBounding = true;

        int N = 1000;

        MultiVariateTimeSeries[] trainSamples = TimeSeriesLoader.loadMultivariateDataset(train, num_sources, segment_length);
        //MultiVariateTimeSeries[] trainSamples = TimeSeriesLoader.createMultivariateDataset(N, num_sources, segment_length);

        TimeSeries[][] trainSamplesSplited = splitMultiDimTimeSeries(num_sources, trainSamples);


        long startTime = System.currentTimeMillis();
        SFA[] sfaIntances = new SFA[num_sources];


        for (int idSource = 0; idSource < num_sources; idSource++) {
            sfaIntances[idSource] = new SFA(SFA.HistogramType.EQUI_DEPTH);
            sfaIntances[idSource].fitWindowing(trainSamplesSplited[idSource], windowLenght, wordLenght, symbols, normMean, lowerBounding);
        }

        final int dimensionality = trainSamples[0].getDimensions();
        final byte usedBits = (byte) Classifier.Words.binlog(symbols);
        final int mask = (1 << (usedBits * wordLenght)) - 1;

        int[][][] words = new int[trainSamples.length][num_sources][];
        Dictionary dict =  new Dictionary();


        for(int i_sample =0; i_sample < trainSamples.length; i_sample++){

            for (int idSource = 0; idSource < num_sources; idSource++) {
                words[i_sample][idSource] = sfaIntances[idSource].transformWindowingInt(trainSamples[i_sample].getTimeSeriesOfOneSource(idSource), wordLenght);
            }
        }
        //bag-of-patterns
        List<BOSSMDStackVS.BagOfPattern> bagOfPatternsTrain = new ArrayList<BOSSMDStackVS.BagOfPattern>(trainSamples.length);


        // iterate all samples
        for (int i_sample = 0; i_sample < trainSamples.length; i_sample++) {

            BOSSMDStackVS.BagOfPattern bop = new BOSSMDStackVS.BagOfPattern(100, trainSamples[i_sample].getLabel());
            for (int idSource = 0; idSource < dimensionality; idSource++) {

                int lastInt = Integer.MIN_VALUE;

                for (int offset = 0; offset < words[i_sample][idSource].length; offset++) {


                    MDWord word = new MDWord(idSource, words[i_sample][idSource][offset] & mask);
                    int dict_v = dict.getWord(word);
                    if (dict_v != lastInt) {
                        bop.bag.putOrAdd(dict_v, 1, 1);
                        lastInt = dict_v;
                    }

                }
            }
            bagOfPatternsTrain.add(bop);
        }

        //get matriz tf-idf
        BOSSMDStackVS bossmd = new BOSSMDStackVS(wordLenght,symbols,windowLenght,true);
        Set<Double> uniqueLabels = uniqueClassLabels(trainSamples);

        ObjectObjectHashMap<Double, IntFloatHashMap> idf = bossmd.createTfIdf(bagOfPatternsTrain.toArray(new BOSSMDStackVS.BagOfPattern[]{}), uniqueLabels);

        System.out.println("\tTime Train: \t" + (System.currentTimeMillis() - startTime) / 1000.0 + " s");



        MultiVariateTimeSeries[] testSamples = TimeSeriesLoader.loadMultivariateDataset(test, num_sources, segment_length);
        words = new int[testSamples.length][num_sources][];
        //Feature Extracting
        startTime = System.currentTimeMillis();


        // create sliding windows -> words

        for(int i_sample =0; i_sample < testSamples.length; i_sample++){

            for (int idSource = 0; idSource < num_sources; idSource++) {
                words[i_sample][idSource] = sfaIntances[idSource].transformWindowingInt(testSamples[i_sample].getTimeSeriesOfOneSource(idSource), wordLenght);
            }
        }

        List<BOSSMDStackVS.BagOfPattern> bagOfPatternsTest = new ArrayList<BOSSMDStackVS.BagOfPattern>(testSamples.length);



        // iterate all samples
        for (int i_sample = 0; i_sample < testSamples.length; i_sample++) {

            BOSSMDStackVS.BagOfPattern bop = new BOSSMDStackVS.BagOfPattern(100, testSamples[i_sample].getLabel());
            for (int idSource = 0; idSource < dimensionality; idSource++) {

                int lastInt = Integer.MIN_VALUE;

                for (int offset = 0; offset < words[i_sample][idSource].length; offset++) {


                    MDWord word = new MDWord(idSource, words[i_sample][idSource][offset] & mask);
                    int dict_v = dict.getWord(word);
                    if (dict_v != lastInt) {
                        bop.bag.putOrAdd(dict_v, 1, 1);
                        lastInt = dict_v;
                    }

                }
            }
            bagOfPatternsTest.add(bop);
        }



        System.out.println("\tTime Extract Features: \t" + (System.currentTimeMillis() - startTime) / 1000.0 + " s");

        BOSSMDStackVS.BagOfPattern[] bop = bagOfPatternsTest.toArray(new BOSSMDStackVS.BagOfPattern[]{});
        //Classification
        startTime = System.currentTimeMillis();

        Classifier.Predictions p = new Classifier.Predictions(new Double[bop.length], 0);
        Double[] goldLabels = new Double[bop.length];
        for (int i_sample = 0; i_sample < bop.length; i_sample++) {

            double bestDistance = 0.0;

            // for each class
            for (ObjectObjectCursor<Double, IntFloatHashMap> classEntry : idf) {

                Double label = classEntry.key;
                IntFloatHashMap stat = classEntry.value;

                // determine cosine similarity
                double distance = 0.0;
                for (IntIntCursor wordFreq : bop[i_sample].bag) {
                    double wordInBagFreq = wordFreq.value;
                    double value = stat.get(wordFreq.key);
                    distance += wordInBagFreq * (value + 1.0);
                }

                // norm by magnitudes
                //if (normMagnitudes) {
                //    distance /= magnitude(stat.values());
                //}

                // update nearest neighbor
                if (distance > bestDistance) {
                    bestDistance = distance;
                    p.labels[i_sample] = label;
                }
            }

            // check if the prediction is correct
            if (compareLabels(bop[i_sample].label, p.labels[i_sample])) {
                p.correct.incrementAndGet();
            }

            goldLabels[i_sample] = bop[i_sample].label;
        }
        p.goldLabels = goldLabels;

        System.out.println("\tClassification: \t" + (System.currentTimeMillis() - startTime) / 1000.0 + " s");

       //System.out.println(p.getConfusionMatrix().toStringProbabilistic());
        System.out.println(p.getConfusionMatrix().printNiceResults());

    }
    public static class MDWord {
        int dim = 0;
        int word = 0;

        public MDWord(int dim, int word) {
            this.dim = dim;
            this.word = word;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            MDWord museWord = (MDWord) o;
            return dim == museWord.dim &&
                    word == museWord.word;
        }

        @Override
        public int hashCode() {
            int result = 1;
            result = 31 * result + Integer.hashCode(word);
            result = 31 * result + Integer.hashCode(dim);
            return result;
        }
    }
    public static class Dictionary {
        ObjectIntHashMap<MDWord> dict;


        public Dictionary() {
            this.dict = new ObjectIntHashMap<MDWord>();

        }

        public void reset() {
            this.dict = new ObjectIntHashMap<MDWord>();

        }

        public int getWord(MDWord word) {
            int index = 0;
            int newWord = -1;
            if ((index = this.dict.indexOf(word)) > -1) {
                newWord = this.dict.indexGet(index);
            } else {
                newWord = this.dict.size() + 1;
                this.dict.put(word, newWord);
            }
            return newWord;
        }


        public int size() {

            return this.dict.size();

        }

    }
    protected static Set<Double> uniqueClassLabels(MultiVariateTimeSeries[] ts) {
        Set<Double> labels = new HashSet<>();
        for (MultiVariateTimeSeries t : ts) {
            labels.add(t.getLabel());
        }
        return labels;
    }
    protected static boolean compareLabels(Double label1, Double label2) {
        // compare 1.0000 to 1.0 in String returns false, hence the conversion to double
        return label1 != null && label2 != null && label1.equals(label2);
    }

}
