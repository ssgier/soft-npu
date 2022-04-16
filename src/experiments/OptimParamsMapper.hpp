#pragma once

#include <Aliases.hpp>

namespace soft_npu {

class OptimParamsMapper {
public:
    explicit OptimParamsMapper(std::shared_ptr<const ParamsType> templateParams);

    SizeType getNumDimensions() const;
    std::vector<double> getInitialValues() const;
    std::shared_ptr<ParamsType> getParams(const double* values) const;
private:
    struct OptimParamsDimInfo {

        OptimParamsDimInfo(std::string path, double floorValue, double ceilValue);

        const std::string path;
        const double floorValue;
        const double ceilValue;
    };

    std::shared_ptr<const ParamsType> templateParams;
    std::vector<OptimParamsDimInfo> dimInfos;

    double extractInitialValue(const OptimParamsDimInfo& dimInfo) const;

    void applyValue(ParamsType& params, const OptimParamsDimInfo& dimInfo, double value) const;
};

}




