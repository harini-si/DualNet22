import torch


class Memory:
    def __init__(self, args, nc_per_task, compute_offsets):
        self.args = args
        self.nc_per_task = nc_per_task
        self.compute_offsets = compute_offsets

        self.n_memories = args.n_memories
        self.mem_cnt = 0
        self.memx = torch.FloatTensor(args.n_tasks, self.n_memories, *args.xdim).to(
            args.device
        )
        self.memy = torch.LongTensor(args.n_tasks, self.n_memories, *args.ydim).to(
            args.device
        )
        self.mem_feat = torch.FloatTensor(
            args.n_tasks, self.n_memories, self.nc_per_task
        ).to(args.device)
        self.bsz = args.batch_size

    def consolidation(self, task):
        t = torch.randint(0, task, (self.bsz,))
        x = torch.randint(0, self.n_memories, (self.bsz,))
        offsets = torch.tensor([self.compute_offsets(i) for i in t])
        xx = self.memx[t, x]
        yy = self.memy[t, x] - offsets[:, 0].to(self.args.device)
        feat = self.mem_feat[t, x]
        mask = torch.zeros(self.bsz, self.nc_per_task)
        for j in range(self.bsz):
            mask[j] = torch.arange(offsets[j][0], offsets[j][1])
        return (
            xx.to(self.args.device),
            yy.to(self.args.device),
            feat.to(self.args.device),
            mask.long().to(self.args.device),
        )
