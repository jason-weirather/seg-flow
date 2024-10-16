import tifffile
import xml.etree.ElementTree as ET

class OMETiffHelper:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    def __init__(self, path):
        """
        Initialize the OMETiffHelper class with the given OME-TIFF file path.
        
        Parameters:
        - path: Path to the OME-TIFF file.
        """
        self.path = path
        self.tif = tifffile.TiffFile(path)
        self.channels_info = self._extract_channel_info()
        self.image_info = self._extract_image_info()

    def _extract_channel_info(self):
        """
        Extract channel information from the OME-TIFF metadata, including channel ID and name.
        Returns a list of dictionaries containing channel details.
        """
        metadata = self.tif.ome_metadata
        channels_info = []
        if metadata:
            root = ET.fromstring(metadata)
            namespaces = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}

            for image in root.findall('ome:Image', namespaces):
                pixels = image.find('ome:Pixels', namespaces)
                for channel in pixels.findall('ome:Channel', namespaces):
                    channel_id = channel.get('ID')
                    channel_name = channel.get('Name') if channel.get('Name') else "Unnamed Channel"
                    channels_info.append({'ID': channel_id, 'Name': channel_name})
        return channels_info

    def _extract_image_info(self):
        """
        Extract general image information such as dimensions, axes type, data type, and pixel size (X and Y).
        Returns a dictionary containing the image information.
        """
        if self.tif.series:
            series = self.tif.series[0]
            shape = series.shape
            dtype = series.dtype
            axes = series.axes
            metadata = self.tif.ome_metadata
            pixel_size_x = None
            pixel_size_y = None
            pixel_size_unit = None
            if metadata:
                root = ET.fromstring(metadata)
                namespaces = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                pixels = root.find('.//ome:Pixels', namespaces)
                if pixels is not None:
                    pixel_size_x = pixels.get('PhysicalSizeX')
                    pixel_size_y = pixels.get('PhysicalSizeY')
                    pixel_size_unit = pixels.get('PhysicalSizeXUnit')
            return {
                'Dimensions': shape,
                'Axes': axes,
                'Data Type': dtype,
                'Pixel Size X': pixel_size_x,
                'Pixel Size Y': pixel_size_y,
                'Pixel Size Unit': pixel_size_unit
            }
        return None

    def get_channel_data_by_index(self, channel_index):
        """
        Get the image data for the specified channel index.
        
        Parameters:
        - channel_index: Index of the channel to extract.
        
        Returns:
        - Numpy array representing the specified channel.
        """
        from .full_image import ContinuousSingleChannelImage
        series = self.tif.series[0]
        return ContinuousSingleChannelImage(series.pages[channel_index].asarray())

    def get_channel_data_by_id(self, channel_id):
        """
        Get the image data for the specified channel by its ID.
        
        Parameters:
        - channel_id: ID of the channel to extract.
        
        Returns:
        - Numpy array representing the specified channel.
        """
        # Find the index of the channel by matching the ID
        for idx, channel in enumerate(self.channels_info):
            if channel['ID'] == channel_id:
                return self.get_channel_data_by_index(idx)
        
        raise ValueError(f"Channel with ID '{channel_id}' not found.")

    def __str__(self):
        """
        Generate a string representation for printing channel and image information.
        
        Returns:
        - String with formatted details about the image.
        """
        if not self.channels_info:
            return "No OME metadata found in the TIFF file."
        info = [f"OME-TIFF File: {self.path}",
                f"Image Dimensions: {self.image_info['Dimensions']}",
                f"Axes: {self.image_info['Axes']}",
                f"Data Type: {self.image_info['Data Type']}",
                f"Pixel Size X: {self.image_info['Pixel Size X']} {self.image_info['Pixel Size Unit']}",
                f"Pixel Size Y: {self.image_info['Pixel Size Y']} {self.image_info['Pixel Size Unit']}",
                f"Number of Channels: {len(self.channels_info)}"]
        for idx, channel in enumerate(self.channels_info, start=0):
            info.append(f"  Channel index {idx}: {channel['Name']} (ID: {channel['ID']})")
        return "\n".join(info)

    def _repr_html_(self):
        """
        Generate an HTML representation for displaying image information in Jupyter Notebook.
        
        Returns:
        - HTML-formatted string with details about the image.
        """
        if not self.channels_info:
            return "<p>No OME metadata found in the TIFF file.</p>"
        html = [f"<h4>OME-TIFF File: {self.path}</h4>",
                f"<p>Image Dimensions: {self.image_info['Dimensions']}</p>",
                f"<p>Axes: {self.image_info['Axes']}</p>",
                f"<p>Data Type: {self.image_info['Data Type']}</p>",
                f"<p>Pixel Size X: {self.image_info['Pixel Size X']} {self.image_info['Pixel Size Unit']}</p>",
                f"<p>Pixel Size Y: {self.image_info['Pixel Size Y']} {self.image_info['Pixel Size Unit']}</p>",
                f"<p>Number of Channels: {len(self.channels_info)}</p>",
                "<ul>"]
        for channel in self.channels_info:
            html.append(f"<li>{channel['Name']} (ID: {channel['ID']})</li>")
        html.append("</ul>")
        return "".join(html)

    def close(self):
        """
        Close the TIFF file to free resources.
        """
        self.tif.close()
