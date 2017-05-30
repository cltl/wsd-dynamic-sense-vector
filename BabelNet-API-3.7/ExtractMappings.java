import java.util.*;
import it.uniroma1.lcl.babelnet.*;
import it.uniroma1.lcl.babelnet.iterators.*;

public class ExtractMappings {

    public static void main(String[] args) {
    	int multipleWnIdsCount = 0;
        BabelNet bn = BabelNet.getInstance();
        for (BabelSynsetIterator it = bn.getSynsetIterator(); it.hasNext(); ) {
        	BabelSynset bnSynset = it.next();
        	List<WordNetSynsetID> wnOffsets = bnSynset.getWordNetOffsets();
        	if (!wnOffsets.isEmpty()) {
	        	System.out.print(bnSynset.getId().getID());
	        	System.out.print("\t");
	        	Set<String> wnIds = new HashSet<>();
	        	for (WordNetSynsetID wnOffset : wnOffsets) {
	        		wnIds.add(wnOffset.getID());
	        	}
	        	for (String wnId : wnIds) {
	        		System.out.print(wnId);
	        		System.out.print(" ");
	        	}
	        	System.out.println();
	        	if (wnIds.size() > 1) {
	        		multipleWnIdsCount += 1;
	        	}
        	}
        }
        System.err.println("Done!");
        System.err.format("Detected %d mappings from one BabelNet synset to "
        		+ "multiple WordNet synsets.\n", multipleWnIdsCount);
    }

}
