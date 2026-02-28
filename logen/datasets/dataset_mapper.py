from logen.datasets.dataset_nuscenes import NuscenesObjectsDataModule
from logen.datasets.dataset_shapenet import ShapeNetObjectsDataModule
from logen.datasets.dataset_kitti360 import Kitti360ObjectsDataModule
from logen.datasets.dataset_sloper4d import Sloper4DObjectsDataModule

dataloaders = {
    'nuscenes': NuscenesObjectsDataModule,
    'shapenet': ShapeNetObjectsDataModule,
    'kitti360': Kitti360ObjectsDataModule,
    'sloper4d': Sloper4DObjectsDataModule
}
