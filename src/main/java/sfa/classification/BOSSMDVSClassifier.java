// Copyright (c) 2017 - Kevin Quispe (kgmq@icomp.ufam.edu.br)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import com.carrotsearch.hppc.IntFloatHashMap;
import com.carrotsearch.hppc.ObjectObjectHashMap;
import com.carrotsearch.hppc.cursors.IntIntCursor;
import com.carrotsearch.hppc.cursors.ObjectObjectCursor;
import sfa.timeseries.MultiDimTimeSeries;
import sfa.transformation.BOSSMD.BagOfPatternMD;
import sfa.transformation.BOSSMD;
import sfa.transformation.BOSSMDVS;


import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The Bags-of-SFA-Symbols MultiDimensional in Vector Space classifier as published in
 * <p>
 * Quispe, Kevin G.: A Multidimensional Symbolic Repsentation of Time Series for Activity Recognition Systems. (201X)
 */
public class BOSSMDVSClassifier extends MDClassifier {

  // default training parameters
  public static double factor = 0.95;

  public static int maxF = 8;
  public static int minF = 8;
  public static int maxS = 4;

  public static boolean normMagnitudes = false;

  // the trained weasel
  public Ensemble<BossMDModel<IntFloatHashMap>> model;

  public BOSSMDVSClassifier() {
    super();
  }


  public static class BossMDModel<E> extends Model {

    public BossMDModel(
            boolean normed,
            int windowLength) {
      super("BOSSMD", -1, 1, -1, 1, normed, windowLength);
    }

    // The inverse document frequencies learned by training
    public ObjectObjectHashMap<Double, E> idf;

    // the trained BOSS MD transformation
    public BOSSMDVS bossmd;

    // the best number of Fourier values to be used
    public int features;
  }


  @Override
  public Score eval(
          final MultiDimTimeSeries[] trainSamples, final MultiDimTimeSeries[] testSamples) {
    long startTime = System.currentTimeMillis();

    Score score = fit(trainSamples);

    if (DEBUG) {
      System.out.println(score.toString());
      outputResult((int) score.training, startTime, testSamples.length);
      System.out.println("");
    }

    // Classify: testing score
    int correctTesting = score(testSamples).correct.get();

    return new Score(
            "BOSSMD",
            correctTesting, testSamples.length,
            score.training, trainSamples.length,
            score.windowLength);
  }

  @Override
  public Score fit(final MultiDimTimeSeries[] trainSamples) {
    // generate test train/split for cross-validation
    generateIndices(trainSamples);

    Score bestScore = null;
    int bestCorrectTraining = 0;

    int minWindowLength = 10;
    int maxWindowLength = getMax(trainSamples, MAX_WINDOW_LENGTH);

    // equi-distance sampling of windows
    ArrayList<Integer> windows = new ArrayList<>();
    double count = Math.sqrt(maxWindowLength);
    double distance = ((maxWindowLength - minWindowLength) / count);
    for (int c = minWindowLength; c <= maxWindowLength; c += distance) {
      windows.add(c);
    }

    for (boolean normMean : NORMALIZATION) {
      // train the shotgun models for different window lengths
      Ensemble<BossMDModel<IntFloatHashMap>> model = fitEnsemble(
              windows.toArray(new Integer[]{}), normMean, trainSamples);
      Double[] labels = predict(model, trainSamples);
      Predictions pred = evalLabels(trainSamples, labels);

      if (bestCorrectTraining <= pred.correct.get()) {
        bestCorrectTraining = pred.correct.get();
        bestScore = model.getHighestScoringModel().score;
        bestScore.training = pred.correct.get();
        this.model = model;
      }
    }

    // return score
    return bestScore;
  }


  @Override
  public Predictions score(final MultiDimTimeSeries[] testSamples) {
    Double[] labels = predict(testSamples);
    return evalLabels(testSamples, labels);
  }


  @Override
  public Double[] predict(final MultiDimTimeSeries[] testSamples) {
    return predict(this.model, testSamples);
  }


  protected Ensemble<BossMDModel<IntFloatHashMap>> fitEnsemble(Integer[] windows,
                                                               boolean normMean,
                                                               MultiDimTimeSeries[] samples) {

    final List<BossMDModel<IntFloatHashMap>> results = new ArrayList<>(windows.length);
    final AtomicInteger correctTraining = new AtomicInteger(0);

    ParallelFor.withIndex(exec, threads, new ParallelFor.Each() {
      Set<Double> uniqueLabels = uniqueClassLabels(samples);

      @Override
      public void run(int id, AtomicInteger processed) {
        for (int i = 0; i < windows.length; i++) {
          if (i % threads == id) {
            BossMDModel<IntFloatHashMap> model = new BossMDModel<>(normMean, windows[i]);
            try {
              BOSSMDVS bossmd = new BOSSMDVS(maxF, maxS, windows[i], model.normed);
              int[][] words = bossmd.createMDWords(samples);

              for (int f = minF; f <= Math.min(model.windowLength, maxF); f += 2) {
                BagOfPatternMD[] bag = bossmd.createBagOfPattern(words, samples, f);

                // cross validation using folds
                int correct = 0;
                for (int s = 0; s < folds; s++) {
                  // calculate the tf-idf for each class
                  ObjectObjectHashMap<Double, IntFloatHashMap> idf = bossmd.createTfIdf(bag,
                          BOSSMDVSClassifier.this.trainIndices[s], this.uniqueLabels);

                  correct += predict(BOSSMDVSClassifier.this.testIndices[s], bag, idf).correct.get();
                }
                if (correct > model.score.training) {
                  model.score.training = correct;
                  model.features = f;

                  if (correct == samples.length) {
                    break;
                  }
                }
              }

              // obtain the final matrix
              BagOfPatternMD[] bag = bossmd.createBagOfPattern(words, samples, model.features);

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


  protected Predictions predict(
          final int[] indices,
          final BagOfPatternMD[] bagOfPatternsTestSamples,
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

  protected Double[] predict(final Ensemble<BossMDModel<IntFloatHashMap>> model, final MultiDimTimeSeries[] testSamples) {
    @SuppressWarnings("unchecked")
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
            final BossMDModel<IntFloatHashMap> score = model.get(i);
            usedLengths.add(score.windowLength);

            BOSSMD model = score.bossmd;

            // create words and BOSS boss for test samples
            int[][] wordsTest = model.createMDWords(testSamples);
            BagOfPatternMD[] bagTest = model.createBagOfPattern(wordsTest, testSamples, score.features);

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

    return score("BOSSMD", testSamples, testLabels, usedLengths);
  }
}