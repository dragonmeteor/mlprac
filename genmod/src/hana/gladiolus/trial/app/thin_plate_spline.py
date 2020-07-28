import math
from tkinter import Tk, Frame, LEFT, Label, BOTH, Button, GROOVE, filedialog, PhotoImage, messagebox, S, W, E, N
from PIL import Image, ImageTk, ImageDraw

import numpy
import torch

from hana.gladiolus.trial.thin_plate_spline_warp import thin_plate_spline_warp
from hana.rindou.util import extract_pytorch_image_from_filelike, rgba_to_numpy_image

IMAGE_SIZE = 512
LANDMARK_GRID_SIZE = 5
MANIPULATING_DISTANCE_THRESHOLD = (10.0 / IMAGE_SIZE * 2) ** 2

MODE_DEFAULT = 0
MODE_MANIPULATING_WARPED_LANDMARK = 1


class ThinPlateSplineApp:
    def __init__(self, master, device: torch.device = torch.device('cpu')):
        self.torch_device = device

        self.master = master
        self.master.title("Thin Plate Spline")

        self.image_frame = Frame(self.master)
        self.image_frame.grid_propagate(1)
        self.image_frame.pack(fill=BOTH)

        pixel = PhotoImage(width=1, height=1)
        self.image_label = Label(self.image_frame, image=pixel, text="Nothing yet!", width=512, height=512)
        self.image_label.grid(row=0)
        self.image_label.bind('<Motion>', self.mouse_motion)
        self.image_label.bind('<Button-1>', self.left_mouse_click)
        self.image_label.bind('<ButtonRelease-1>', self.left_mouse_release)

        self.load_source_image_button = Button(self.image_frame, text="Load Image ...", relief=GROOVE,
                                               command=self.load_source_image)
        self.load_source_image_button.grid(row=1, sticky=N + E + W + S)

        self.source_image = None

        self.landmarks = torch.zeros(LANDMARK_GRID_SIZE * LANDMARK_GRID_SIZE, 2, device=self.torch_device)
        self.warped_landmaks = torch.zeros(LANDMARK_GRID_SIZE * LANDMARK_GRID_SIZE, 2, device=self.torch_device)
        for iy in range(LANDMARK_GRID_SIZE):
            y = -0.75 + 1.5 * iy / (LANDMARK_GRID_SIZE - 1)
            for ix in range(LANDMARK_GRID_SIZE):
                x = -0.75 + 1.5 * ix / (LANDMARK_GRID_SIZE - 1)
                index = iy * LANDMARK_GRID_SIZE + ix
                self.landmarks[index, 0] = x
                self.landmarks[index, 1] = y
                self.warped_landmaks[index, 0] = x
                self.warped_landmaks[index, 1] = y

        self.warped_pil_image = None
        self.needs_update = False
        self.mouse_screen_pos = None
        self.manipulating_warped_landmark = None
        self.mode = MODE_DEFAULT
        self.start_mouse_image_pos = None
        self.start_warped_landmark = None

        self.master.after(1000 // 30, self.update_image)

    def mouse_motion(self, event):
        self.mouse_screen_pos = (event.x, event.y)
        self.mouse_image_pos = ((event.x + 0.5) / IMAGE_SIZE * 2 - 1, (IMAGE_SIZE - event.y + 0.5) / IMAGE_SIZE * 2 - 1)
        if self.mode == MODE_DEFAULT:
            self.update_manipulating_warped_landmark()
        elif self.mode == MODE_MANIPULATING_WARPED_LANDMARK:
            dx = self.mouse_image_pos[0] - self.start_mouse_image_pos[0]
            dy = self.mouse_image_pos[1] - self.start_mouse_image_pos[1]
            self.warped_landmaks[self.manipulating_warped_landmark, 0] = self.start_warped_landmark[0] + dx
            self.warped_landmaks[self.manipulating_warped_landmark, 1] = self.start_warped_landmark[1] + dy
            self.needs_update = True

    def left_mouse_click(self, event):
        if self.source_image is None:
            return
        if self.manipulating_warped_landmark is not None:
            self.mode = MODE_MANIPULATING_WARPED_LANDMARK
            self.start_mouse_image_pos = (self.mouse_image_pos[0], self.mouse_image_pos[1])
            self.start_warped_landmark = (self.warped_landmaks[self.manipulating_warped_landmark, 0].item(),
                                          self.warped_landmaks[self.manipulating_warped_landmark, 1].item())

    def left_mouse_release(self, event):
        self.mode = MODE_DEFAULT

    def update_manipulating_warped_landmark(self):
        nearest_index, nearest_distance = self.find_nearest_warped_landmark(self.mouse_image_pos[0],
                                                                            self.mouse_image_pos[1])
        if nearest_distance <= MANIPULATING_DISTANCE_THRESHOLD:
            self.manipulating_warped_landmark = nearest_index
        else:
            self.manipulating_warped_landmark = None

    def find_nearest_warped_landmark(self, x, y):
        lowest_distance = math.inf
        lowest_index = -1
        for i in range(LANDMARK_GRID_SIZE * LANDMARK_GRID_SIZE):
            dx = self.warped_landmaks[i, 0].item() - x
            dy = self.warped_landmaks[i, 1].item() - y
            distance = dx * dx + dy * dy
            if distance < lowest_distance:
                lowest_distance = distance
                lowest_index = i
        return lowest_index, lowest_distance

    def load_source_image(self):
        file_name = filedialog.askopenfilename(
            filetypes=[("PNG", '*.png')],
            initialdir="d:/me/workspace/hana2/data/gladiolus/_20200208")
        if len(file_name) > 0:
            image = PhotoImage(file=file_name)
            if image.width() != IMAGE_SIZE or image.height() != IMAGE_SIZE:
                message = "The loaded image has size %dx%d, but we require %dx%d." \
                          % (image.width(), image.height(), IMAGE_SIZE, IMAGE_SIZE)
                messagebox.showerror("Wrong image size!", message)
            self.source_image = extract_pytorch_image_from_filelike(file_name) \
                .to(self.torch_device) \
                .unsqueeze(dim=0) \
                .flip(dims=[2])
            self.needs_update = True

    def update_image(self):
        if self.source_image is None:
            self.master.after(1000 // 30, self.update_image)
            return
        if self.needs_update:
            warped_image = thin_plate_spline_warp(self.source_image, self.landmarks, self.warped_landmaks) \
                .detach().cpu()
            numpy_image = rgba_to_numpy_image(warped_image[0])
            self.warped_pil_image = Image.fromarray(numpy.uint8(numpy.rint(numpy_image * 255.0)), mode='RGBA')

        pil_image = self.warped_pil_image.copy()
        draw = ImageDraw.Draw(pil_image)
        for i in range(LANDMARK_GRID_SIZE * LANDMARK_GRID_SIZE):
            p = (self.warped_landmaks[i, :] + 1) / 2 * IMAGE_SIZE
            x = p[0].item()
            y = p[1].item()
            draw.rectangle(((x - 2, y - 2), (x + 2, y + 2)), fill=(255, 0, 0))
        for iy0 in range(LANDMARK_GRID_SIZE):
            iy1 = iy0 + 1
            for ix0 in range(LANDMARK_GRID_SIZE):
                ix1 = ix0 + 1
                i00 = iy0 * LANDMARK_GRID_SIZE + ix0
                p00 = (self.warped_landmaks[i00, :] + 1) / 2 * IMAGE_SIZE
                if ix1 < LANDMARK_GRID_SIZE and iy0 < LANDMARK_GRID_SIZE:
                    i10 = iy0 * LANDMARK_GRID_SIZE + ix1
                    p10 = (self.warped_landmaks[i10, :] + 1) / 2 * IMAGE_SIZE
                    draw.line(((p00[0].item(), p00[1].item()), (p10[0].item(), p10[1].item())), fill=(255, 0, 0))
                if ix0 < LANDMARK_GRID_SIZE and iy1 < LANDMARK_GRID_SIZE:
                    i01 = iy1 * LANDMARK_GRID_SIZE + ix0
                    p01 = (self.warped_landmaks[i01, :] + 1) / 2 * IMAGE_SIZE
                    draw.line(((p00[0].item(), p00[1].item()), (p01[0].item(), p01[1].item())), fill=(255, 0, 0))

        if self.manipulating_warped_landmark is not None:
            index = self.manipulating_warped_landmark
            p = (self.warped_landmaks[index, :] + 1) / 2 * IMAGE_SIZE
            x0 = p[0].item() - 5
            y0 = p[1].item() - 5
            x1 = p[0].item() + 5
            y1 = p[1].item() + 5
            draw.rectangle(((x0, y0), (x1, y1)), fill=None, outline=(255, 0, 0))

        pil_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)
        photo_image = ImageTk.PhotoImage(image=pil_image)
        self.image_label.configure(image=photo_image, text="")
        self.image_label.image = photo_image
        self.image_label.grid(row=0)
        self.needs_update = False

        self.master.after(1000 // 30, self.update_image)


if __name__ == "__main__":
    root = Tk()
    app = ThinPlateSplineApp(master=root)
    root.mainloop()
