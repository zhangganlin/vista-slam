import cv2
import DBoW3Py as dbow

class LoopDetector:
    def __init__(self, vocab_path, loop_dist_min, loop_nms, loop_cand_thresh_neighbor):
        self.vocab = dbow.Vocabulary()
        self.vocab.load(vocab_path)
        self.bow_feats = []
        self.orb = cv2.ORB_create()
        
        self.loop_dist_min = loop_dist_min     # if node i has a loop edge to node j, then |i-j| > loop_dist_min
        self.loop_nms = loop_nms          # non-maximum suppression for loop closure
        self.loop_cand_thresh_neighbor = loop_cand_thresh_neighbor  # the loop candidate should have similarity at least larger than x neighbor
    def compute_bow_feat(self, image):
        _, descriptors = self.orb.detectAndCompute(image, None) 
        if descriptors is None:
            self.bow_feats.append(None)
            return None
        bow_vector = self.vocab.transform(descriptors)
        self.bow_feats.append(bow_vector)
        return bow_vector
    
    def detect_loop(self, image, farthest_neighbor):
        bow_feat_i = self.compute_bow_feat(image)
        i = len(self.bow_feats) - 1 
        loop_farthest_neighbor = max(0,i-self.loop_cand_thresh_neighbor)
        bow_feat_i = self.bow_feats[i]
        
        neighbor_sims = []
        for j in range(loop_farthest_neighbor, i):
            if bow_feat_i is None or self.bow_feats[j] is None:
                continue
            similarity =self.vocab.score(bow_feat_i, self.bow_feats[j])
            neighbor_sims.append(similarity)

        sim_thresh = 1.0 if len(neighbor_sims)==0 \
                     else min(neighbor_sims)
        last_egde = farthest_neighbor
        loop_candi_list = []
        for j in reversed(range(0, farthest_neighbor)):
            if last_egde - j > self.loop_nms and i-j > self.loop_dist_min:
                if bow_feat_i is None or self.bow_feats[j] is None:
                    continue
                similarity = self.vocab.score(bow_feat_i, self.bow_feats[j])
                if similarity > sim_thresh:
                    loop_candi_list.append((j, similarity))
                    last_egde = j
        loop_candi_list = sorted(loop_candi_list,key=lambda x: x[1],reverse=True)
        
        return loop_candi_list
