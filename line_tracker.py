
import limap.base as _base
import limap.util.io as limapio
import limap.visualize as limapvis

import numpy as np

#class Window:
#    def __init__(self, bounding_box, window_id=None):
#        self.bounding_box = bounding_box  # Format: (x_min, y_min, x_max, y_max)
#        self.id = window_id

class LineTracker:
    def __init__(self, finaltracks):
        self.imagecols = _base.ImageCollection(limapio.read_npy(finaltracks+"/imagecols.npy").item())
        print(f"NumCameras = {self.imagecols.NumCameras()}")
        print(f"NumImages = {self.imagecols.NumImages()}")
        #for img_id in range(1, self.imagecols.NumCameras()+1):
        #    print(self.get_image_name(img_id))
                #print(f"camimage = {imagecols.camimage(img_id)}")
                #print(f"campose = {imagecols.campose(img_id)}")

        # lines and detections
        self.lines, self.linetracks = limapio.read_lines_from_input(finaltracks)
        # limap.base.LineTrack: Associated line track across multi-view.
        print(f"Lines = {len(self.lines)}")
        print(f"LineTracks = {len(self.linetracks)}")

    def get_number_of_cameras(self):
        return self.imagecols.NumCameras()

    def get_number_of_images(self):
        return self.imagecols.NumImages()

    def get_num_tracks(self):
        return len(self.linetracks)

    def get_image_name(self, img_id):
        if self.imagecols.exist_image(img_id):
            return self.imagecols.image_name(img_id)
        return None

    def get_2d_line_in_image(self, track_id, img_id):
        if self.linetracks[track_id].HasImage(img_id):
            if img_id in self.linetracks[track_id].image_id_list:
                image_index = self.linetracks[track_id].image_id_list.index(img_id)
                return self.linetracks[track_id].line2d_list[image_index]
        return None

    def is_track_in_image(self, track_id, img_id):
        return self.linetracks[track_id].HasImage(img_id)

    def get_3d_line(self, track_id):
        return self.linetracks[track_id].line.as_array()

    def get_a_projection(self, track_id, image_id):
        camview = self.imagecols.camview(image_id)
        line = self.linetracks[track_id].line
        return line.projection(camview)

    def get_campose(self, img_id):
        campose = self.imagecols.campose(img_id)
        R = np.array(campose.R())  # Convert to NumPy array if not already
        t = np.array(campose.T()).reshape(-1, 1)  # Ensure t is a column vector
        return np.hstack((R, t))

    def get_intrinsic_matrix(self, img_id):
        camview = self.imagecols.camview(img_id)
        K = np.array(camview.K())
        return K[0,0], K[1,1], K[0,2], K[1,2]
        

