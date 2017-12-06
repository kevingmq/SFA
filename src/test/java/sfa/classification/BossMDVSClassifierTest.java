package sfa.classification;

import java.util.ArrayList;
import java.util.List;

public class BossMDVSClassifierTest extends AbstractClassifierMDTest{
    @Override
    protected List<DataSet> getDataSets() {
        List<DataSet> dataSets=new ArrayList<>();
        //Coffee;BOSS VS;1.0;1.0
        dataSets.add(new DataSet("Coffee", 1.0, 1.0));
        //Beef;BOSS VS;1.0;0.833
        dataSets.add(new DataSet("Beef", 1.0, 0.833));
        //CBF;BOSS VS;1.0;0.998
        dataSets.add(new DataSet("CBF", 1.0, 0.998));
        return dataSets;
    }

    @Override
    protected MDClassifier initClassifier() {
        return new BOSSMDVSClassifier();
    }
}
