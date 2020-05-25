import numpy as np
from typing import List
from pathlib import Path


# This class is made for personal use and is not intended to
# perfectly parse any valid .pcd file.
class PCDCloud:
    def __init__(self, data, fields, counts=None, sizes=None, types=None):
        self.data = data
        self.fields = fields
        self.counts = [1 for _ in fields] if counts is None else counts
        self.sizes = [4 for _ in fields] if sizes is None else sizes
        self.types = ['F' for _ in fields] if types is None else types

    @classmethod
    def from_file(cls, filepath: Path):
        with open(filepath, 'rb') as f:
            lines = f.readlines()
            points = -1

            fields = []
            counts = []
            types = []
            sizes = []
            binary = False

            for i, line in enumerate(lines):
                line = line.decode('utf-8').strip('\n')
                if i in range(2):
                    continue
                elif i == 2:
                    fields = line.split(' ')[1:]
                elif i == 3:
                    sizes = [int(x) for x in line.split(' ')[1:]]
                elif i == 4:
                    types = line.split(' ')[1:]
                elif i == 5:
                    counts = [int(x) for x in line.split(' ')[1:]]
                elif i in range(6, 9):
                    continue  # Do nothing for now
                elif i == 9:
                    # Don't store point count as member
                    points = int(line.split(' ')[1])
                    # so we don't have to deal with changing it when points are
                    # added
                elif i == 10:
                    binary = line == "DATA binary"
                    break
            if not binary:
                raise NotImplementedError("Only support binary data for now")
            else:
                non_padding_columns = []
                non_padding_index = []
                if '_' in fields:
                    # File contains padding data from CloudCompare :(
                    current_count = 0
                    for i in range(len(fields)):
                        if fields[i] == '_':
                            sizes[i] = 4
                            counts[i] //= 4
                        else:
                            non_padding_columns.extend(
                                range(current_count, current_count + counts[i]))
                            non_padding_index.append(i)

                        current_count += counts[i]

                binary_data = b''.join(lines[11:])
                n_columns = sum(counts)
                data = np.fromstring(
                    binary_data,
                    dtype=np.float32,
                    count=n_columns * points)
                data = data.reshape((points, n_columns))

                if '_' in fields:
                    data = data[:, non_padding_columns]
                    fields = [fields[i] for i in non_padding_index]
                    types = [types[i] for i in non_padding_index]
                    counts = [counts[i] for i in non_padding_index]
                    sizes = [sizes[i] for i in non_padding_index]

                return cls(data, fields, counts, sizes, types)

    @classmethod
    def create_similar_empty(cls, cloud):
        return cls(np.empty((0, cloud.data.shape[1])),
                   cloud.fields,
                   cloud.counts,
                   cloud.sizes,
                   cloud.types)

    def point_count(self):
        return self.data.shape[0]

    def add_fields(self, field_names: List[str]) -> None:
        self.fields.extend(field_names)
        for _ in field_names:
            self.counts.append(1)
            self.sizes.append(4)
            self.types.append("F")
        self.data = np.hstack((self.data,
                               np.zeros((self.data.shape[0], len(field_names)),
                                        dtype=np.float32)))

    def save(self, filename: Path, binary=True) -> None:
        header = "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
        header += "FIELDS " + ' '.join(self.fields) + '\n'
        header += "SIZE " + ' '.join([str(x) for x in self.sizes]) + '\n'
        header += "TYPE " + ' '.join(self.types) + '\n'
        header += "COUNT " + ' '.join([str(x) for x in self.counts]) + '\n'
        header += "WIDTH " + str(self.point_count()) + '\n'
        header += "HEIGHT 1\n"
        header += "VIEWPOINT 0 0 0 1 0 0 0\n"  # TODO
        header += "POINTS " + str(self.point_count()) + '\n'
        if binary:
            header += "DATA binary\n"
            with open(filename, 'wb') as f:
                f.write(header.encode('utf-8'))
                f.write(self.data.tobytes())
        else:
            raise NotImplementedError("ASCII save TODO!")

    def field_to_columns(self, field: str) -> List[int]:
        field_index = -1
        for i, f in enumerate(self.fields):
            if f == field:
                field_index = i
        if field_index == -1:
            raise ValueError(
                f"Field {field} not present in loaded point cloud.")
        return [sum(self.counts[:field_index]) +
                i for i in range(self.counts[field_index])]

    def fields_to_columns(self, fields: List[str]) -> List[int]:
        result = []
        for field in fields:
            result.extend(self.field_to_columns(field))
        return result

    def get_field_view(self, field: str):
        return self.data[:, self.field_to_columns(field)]

    def get_fields_view(self, fields: List[str]):
        return self.data[:, self.fields_to_columns(fields)]

    def get_xyz_view(self):
        return self.get_fields_view(['x', 'y', 'z'])

    def get_xyz_normals_view(self):
        return self.get_fields_view(['x', 'y', 'z',
                                     'normal_x', 'normal_y', 'normal_z'])

    def get_normals_view(self):
        return self.get_fields_view(['normal_x', 'normal_y', 'normal_z'])

    def get_xy_view(self):
        return self.get_fields_view(['x', 'y'])
