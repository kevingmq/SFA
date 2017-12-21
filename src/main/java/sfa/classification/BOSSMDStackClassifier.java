package sfa.classification;

import com.carrotsearch.hppc.IntFloatHashMap;
import com.carrotsearch.hppc.ObjectObjectHashMap;
import com.carrotsearch.hppc.cursors.IntIntCursor;
import com.carrotsearch.hppc.cursors.ObjectObjectCursor;
import sfa.timeseries.MultiVariateTimeSeries;
import sfa.timeseries.TimeSeries;
import sfa.transformation.BOSSMDStackVS;
import sfa.transformation.BOSSMDStackVS.BagOfPattern;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

public class BOSSMDStackClassifier extends Classifier{

    // default training parameters
    public static double factor = 0.90;

    public static int maxF = 16;
    public static int minF = 4;
    public static int maxS = 4;

    public static boolean normMagnitudes = false;

    // the trained weasel
    public Ensemble<BOSSMDStackModel<IntFloatHashMap>> model;

    public BOSSMDStackClassifier(){
        super();
    }

    public static class BOSSMDStackModel<E> extends Model {

        public BOSSMDStackModel(
                boolean normed,
                int windowLength) {
            super("BOSSMDStackModel", -1, 1, -1, 1, normed, windowLength);
        }

        // The inverse document frequencies learned by training
        public ObjectObjectHashMap<Double, E> idf;

        // the trained BOSS MD transformation
        public BOSSMDStackVS bossmd;

        // the best number of Fourier values to be used
        public int features;
    }


    @Override
    public Score eval(
            final TimeSeries[] trainSamples, final TimeSeries[] testSamples) {
        throw new RuntimeException("Please use: eval(" +
                "final MultiVariateTimeSeries[] trainSamples, final MultiVariateTimeSeries[] testSamples)");
    }

    @Override
    public Double[] predict(TimeSeries[] samples) {
        throw new RuntimeException("Please use: predict(final MultiVariateTimeSeries[] samples)");
    }

    @Override
    public Score fit(final TimeSeries[] trainSamples) {
        throw new RuntimeException("Please use: fit(final MultiVariateTimeSeries[] trainSamples)");
    }

    @Override
    public Predictions score(final TimeSeries[] testSamples) {
        throw new RuntimeException("Please use: score(final MultiVariateTimeSeries[] testSamples)");
    }


    public Ensemble<BOSSMDStackModel<IntFloatHashMap>> fit(final MultiVariateTimeSeries[] trainSamples) {

        // generate test train/split for cross-validation
        generateIndices(trainSamples);

        Ensemble<BOSSMDStackModel<IntFloatHashMap>> resultFit = null;

        int bestCorrectTraining = 0;

        int minWindowLength = 10;
        int maxWindowLength = getMax(trainSamples,MAX_WINDOW_LENGTH);

        // equi-distance sampling of windows
        ArrayList<Integer> windows = new ArrayList<>();
        double count = Math.sqrt(maxWindowLength);
        double distance = ((maxWindowLength - minWindowLength) / count);
        //distance = 1;
        for (int c = minWindowLength; c <= maxWindowLength; c += distance) {
            windows.add(c);
        }
        long startTime = System.currentTimeMillis();
        for (boolean normMean : NORMALIZATION) {
            // train the shotgun models for different window lengths
            Ensemble<BOSSMDStackModel<IntFloatHashMap>> model = fitEnsemble(windows.toArray(new Integer[]{}), normMean, trainSamples);
            BOSSMDStackModel<IntFloatHashMap> bestModel = model.getHighestScoringModel();
            if (DEBUG) {
                System.out.println("BOSSMDStackVS " + " Training:\t S_" + maxS + " F_" + bestModel.features + "\tw_" + bestModel.windowLength + "\tnormed: \t" + normMean);
                outputResult(bestModel.score.training, startTime, trainSamples.length);
            }

            if (bestCorrectTraining <= bestModel.score.training) {
                bestCorrectTraining = bestModel.score.training;
                resultFit = model;

            }
            startTime = System.currentTimeMillis();
        }

        return resultFit;
    }

    public Predictions score(final MultiVariateTimeSeries[] testSamples) {
        Double[] labels = predict(testSamples);
        return evalLabels(testSamples, labels);
    }


    public Predictions predict(final int[] indices,
                            final BagOfPattern[] bagOfPatternsTestSamples,
                            final ObjectObjectHashMap<Double, IntFloatHashMap> matrixTrain) {

        Predictions p = new Predictions(new Double[bagOfPatternsTestSamples.length], 0);

        ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
            @Override
            public void run(int id, AtomicInteger processed) {
                // iterate each sample to classify
                for (int i : indices) {
                    if (i % BLOCKS == id) {
                        double bestDistance = 0.0;

                        // for each class
                        for (ObjectObjectCursor<Double, IntFloatHashMap> classEntry : matrixTrain) {

                            Double label = classEntry.key;
                            IntFloatHashMap stat = classEntry.value;

                            // determine cosine similarity
                            double distance = 0.0;
                            for (IntIntCursor wordFreq : bagOfPatternsTestSamples[i].bag) {
                                double wordInBagFreq = wordFreq.value;
                                double value = stat.get(wordFreq.key);
                                distance += wordInBagFreq * (value + 1.0);
                            }

                            // norm by magnitudes
                            if (normMagnitudes) {
                                distance /= magnitude(stat.values());
                            }

                            // update nearest neighbor
                            if (distance > bestDistance) {
                                bestDistance = distance;
                                p.labels[i] = label;
                            }
                        }

                        // check if the prediction is correct
                        if (compareLabels(bagOfPatternsTestSamples[i].label, p.labels[i])) {
                            p.correct.incrementAndGet();
                        }
                    }
                }
            }
        });

        return p;
    }

    public Score eval(final MultiVariateTimeSeries[] trainSamples, final MultiVariateTimeSeries[] testSamples) {
        ArrayList<String> output = new ArrayList<>();
        long startTimeFit = System.currentTimeMillis();

        this.model = fit(trainSamples);

        //Prints
        final BOSSMDStackModel<IntFloatHashMap> highestScoringModel = this.model.getHighestScoringModel();

        output.add(String.valueOf((System.currentTimeMillis() - startTimeFit) / 1000.0));

        long startTimePredict = System.currentTimeMillis();
        // Classify: testing score
        Predictions p = score(testSamples);
        output.add(String.valueOf((System.currentTimeMillis() - startTimePredict) / 1000.0));

        int correctTesting = p.correct.get();
        double error = formatError(correctTesting,testSamples.length);

        output.add(String.valueOf(highestScoringModel.bossmd.alphabetSize));
        output.add(String.valueOf(highestScoringModel.features));
        output.add(String.valueOf(highestScoringModel.score.windowLength));
        output.add(String.valueOf(highestScoringModel.normed));

       // output.add(String.valueOf(1-error));
        String outputString = listToCsv(output,',');

        return new Score(
                "BOSS MDStack",
                correctTesting, testSamples.length,
                highestScoringModel.score.training, trainSamples.length,
                highestScoringModel.score.windowLength, outputString ,p.getConfusionMatrix());
    }


    protected String listToCsv(List<String> listOfStrings, char separator) {
        StringBuilder sb = new StringBuilder();

        // all but last
        for(int i = 0; i < listOfStrings.size() - 1 ; i++) {
            sb.append(listOfStrings.get(i));
            sb.append(separator);
        }

        // last string, no separator
        if(listOfStrings.size() > 0){
            sb.append(listOfStrings.get(listOfStrings.size()-1));
        }

        return sb.toString();
    }

    protected Ensemble<BOSSMDStackModel<IntFloatHashMap>> fitEnsemble(Integer[] windows,
                                                                 boolean normMean,
                                                                 MultiVariateTimeSeries[] samples) {

        final List<BOSSMDStackModel<IntFloatHashMap>> results = new ArrayList<>(windows.length);
        final AtomicInteger correctTraining = new AtomicInteger(0);

        ParallelFor.withIndex(exec, threads, new ParallelFor.Each() {
            Set<Double> uniqueLabels = uniqueClassLabels(samples);

            @Override
            public void run(int id, AtomicInteger processed) {
                for (int i = 0; i < windows.length; i++) {
                    if (i % threads == id) {
                        BOSSMDStackModel<IntFloatHashMap> model = new BOSSMDStackModel<>(normMean, windows[i]);
                        try {
                            BOSSMDStackVS bossmd = new BOSSMDStackVS(maxF, maxS, windows[i], model.normed);
                            int[/*samples */][/*source */][/*wordInt*/] words = bossmd.createWords(samples);


                            for (int f = minF; f <= Math.min(model.windowLength, maxF); f += 2) {
                                BagOfPattern[] bag = bossmd.createBagOfPattern(words, samples, f);

                                // cross validation using folds
                                int correct = 0;
                                for (int s = 0; s < folds; s++) {
                                    // calculate the tf-idf for each class
                                    ObjectObjectHashMap<Double, IntFloatHashMap> idf = bossmd.createTfIdf(bag,
                                            BOSSMDStackClassifier.this.trainIndices[s], this.uniqueLabels);
                                    Predictions p = predict(testIndices[s], bag, idf);
                                    correct += p.correct.get();
                                }
                                if (correct > model.score.training) {
                                    model.score.training = correct;
                                    model.score.trainSize = samples.length;
                                    model.features = f;

                                    if (correct == samples.length) {
                                        break;
                                    }
                                }
                            }

                            // obtain the final matrix
                            BagOfPattern[] bag = bossmd.createBagOfPattern(words, samples, model.features);

                            // calculate the tf-idf for each class
                            model.idf = bossmd.createTfIdf(bag, this.uniqueLabels);
                            model.bossmd = bossmd;

                        } catch (Exception e) {
                            e.printStackTrace();
                        }

                        // keep best scores
                        synchronized (correctTraining) {
                            if (model.score.training > correctTraining.get()) {
                                correctTraining.set(model.score.training);
                            }

                            // add to ensemble if train-score is within factor to the best score
                            if (model.score.training >= correctTraining.get() * factor) {
                                results.add(model);
                            }
                        }
                    }
                }
            }
        });

        // returns the ensemble based on the best window-lengths within factor
        return filterByFactor(results, correctTraining.get(), factor);
    }

    protected Double[] predict(final Ensemble<BOSSMDStackModel<IntFloatHashMap>> model, final MultiVariateTimeSeries[] testSamples) {

        final List<Pair<Double, Integer>>[] testLabels = new List[testSamples.length];
        for (int i = 0; i < testLabels.length; i++) {
            testLabels[i] = new ArrayList<>();
        }

        final List<Integer> usedLengths = Collections.synchronizedList(new ArrayList<>(model.size()));
        final int[] indicesTest = createIndices(testSamples.length);

        // parallel execution
        ParallelFor.withIndex(exec, threads, new ParallelFor.Each() {
            @Override
            public void run(int id, AtomicInteger processed) {
                // iterate each sample to classify
                for (int i = 0; i < model.size(); i++) {
                    if (i % threads == id) {
                        final BOSSMDStackModel<IntFloatHashMap> score = model.get(i);
                        usedLengths.add(score.windowLength);

                        BOSSMDStackVS model = score.bossmd;

                        // create words and BOSS boss for test samples
                        int[][][] wordsTest = model.createWords(testSamples);
                        BagOfPattern[] bagTest = model.createBagOfPattern(wordsTest, testSamples, score.features);

                        Predictions p = predict(indicesTest, bagTest, score.idf);

                        for (int j = 0; j < p.labels.length; j++) {
                            synchronized (testLabels[j]) {
                                testLabels[j].add(new Pair<>(p.labels[j], score.score.training));
                            }
                        }
                    }
                }
            }
        });

        return score("BOSS MDStack", testSamples, testLabels, usedLengths);
    }

    public Double[] predict(final MultiVariateTimeSeries[] testSamples) {
        return predict(this.model, testSamples);
    }
}
