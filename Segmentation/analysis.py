import numpy as np
import matplotlib.pyplot as plt

def smooth(loss):
    n = loss.shape[0]
    w = 100

    loss_padded = np.zeros((n+w+w))
    loss_padded[0:w] = loss[0]
    loss_padded[n+w:n+w+w] = loss[-1]
    loss_padded[w:n+w] = loss
    _loss = np.zeros((n+w+w))
    for i in range(w, n+w):
        _loss[i-w] = np.mean(loss_padded[i-w:i+w])
    return _loss

'''
plot training loss
'''
# loss_unet = np.load('./unet-2/loss.npy')
# loss_unet = smooth(loss_unet)
# loss_dilation = np.load('./dilation-3/loss.npy')
# loss_dilation = smooth(loss_dilation)
# loss_spp = np.load('./spp-3/loss.npy')
# loss_spp = smooth(loss_spp)

# plt.figure()
# # plt.plot(np.load('./unet-2/loss.npy'))
# plt.plot(loss_unet)
# plt.plot(loss_dilation)
# plt.plot(loss_spp)
# plt.ylabel('loss')
# plt.xlabel('iteration')
# plt.title('train')
# plt.legend(['unet', '+dilation', '+dilation+spp'])
# plt.savefig('show.png')
# plt.show()

# print(loss_unet.shape)
# print(loss_dilation.shape)
# print(loss_spp.shape)

'''
testing
'''
id = 99
accuracy_unet = np.load('./unet-2-test/accuracy_%d.npy'%id)
print('---------------- unet accuracy ----------------')
print(accuracy_unet.shape)
print(np.round(accuracy_unet, 2))
print(np.mean(accuracy_unet))
print('---------------- unet accuracy ----------------')

accuracy_dilation = np.load('./dilation-3-test/accuracy_%d.npy'%id)
print('---------------- dilation accuracy ----------------')
print(accuracy_dilation.shape)
print(np.round(accuracy_dilation, 2))
print(np.mean(accuracy_dilation))
print('---------------- dilation accuracy ----------------')

accuracy_spp = np.load('./spp-3-test/accuracy_%d.npy'%id)
print('---------------- spp accuracy ----------------')
print(accuracy_spp.shape)
print(np.round(accuracy_spp, 2))
print(np.mean(accuracy_spp))
print('---------------- spp accuracy ----------------')


markdown = '| class | unet | +dilation | +dilation + spp |' + '\n'
markdown += '|:------|:-----|:----------|:----------------|' + '\n'
for i in range(21):
    markdown += '| ' + str(i) + ' | ' + str(accuracy_unet[i]) + ' | ' + str(accuracy_dilation[i]) + ' | ' + str(accuracy_spp[i]) + ' |' + '\n'
    # markdown += '\hline' + '\n'

markdown += '| ' + 'mIoU' + ' | ' + str(np.mean(accuracy_unet)) + ' | ' + str(np.mean(accuracy_dilation)) + ' | ' + str(np.mean(accuracy_spp)) + ' |'

print(markdown)