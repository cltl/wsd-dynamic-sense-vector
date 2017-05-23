import java.util.*;
import it.uniroma1.lcl.babelnet.*;
import it.uniroma1.lcl.babelnet.iterators.*;

public class ExtractMappings {

    public static void main(String[] args) {
        BabelNet bn = BabelNet.getInstance();
        for (BabelSynsetIterator it = bn.getSynsetIterator(); it.hasNext(); ) {
        	BabelSynset bnSynset = it.next();
        	List<WordNetSynsetID> wnOffsets = bnSynset.getWordNetOffsets();
        	if (!wnOffsets.isEmpty()) {
	        	System.out.print(bnSynset.getId().getID());
	        	System.out.print("\t");
	        	for (WordNetSynsetID wnOffset : wnOffsets) {
	        		System.out.print(wnOffset.getID());
	        		System.out.print(" ");
	        	}
	        	System.out.println();
        	}
        }
    }

}
