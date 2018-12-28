# implement a multi-cycle average filter

def avgFilter(data,cyc):
    # data: the data to be filtered
    # cyc: number of cycles in the filter
    filterOutput = []

    data = list(data)
    for i in range(len(data)):

        if i + 1 < cyc:
            d = data[0:i+1]
            tot = sum(d)
            filterOutput.append(tot/len(d))
        else:
            d = data[i+1-cyc:i+1] # i+1 so that it includes the i-th data as well
            tot = sum(d)
            filterOutput.append(tot/len(d))
    return filterOutput

