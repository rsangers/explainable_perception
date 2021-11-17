from nets.rcnn import RCnn


class RankCnn(RCnn):

    def forward(self, images):
        batch_size = images.size()[0]
        features = self.cnn(images)
        x = features.view(batch_size,self.cnn_size[1]*self.cnn_size[2]*self.cnn_size[3])
        x = self.rank_fc_1(x)
        x = self.relu(x)
        x = self.rank_fc_out(x)
        return x


